"""索引流水线：load → chunk → embed → upsert（幂等）。

职责：
- 单文件索引：解析、分块、向量化并写入 Chroma；重复索引先删旧数据
- 目录批量索引：按扩展名过滤，逐文件容错，输出汇总报告
- 来源删除：按 source_id 移除全部向量

source_id 约定：文件绝对路径的 SHA256 前 16 位十六进制，
同一文件路径在不同时间索引得到相同 ID，是实现幂等覆盖的基础。
chunk 唯一 ID 形如 ``{source_id}_{chunk_index}``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rag_mcp_server.chunking import chunk_text
from rag_mcp_server.common.exceptions import EmbeddingError, LoaderError
from rag_mcp_server.config.settings import Settings
from rag_mcp_server.embeddings.base import EmbeddingClient
from rag_mcp_server.loaders import SUPPORTED_EXTENSIONS, load_document
from rag_mcp_server.vectorstore.chroma_store import ChromaStore, VectorRecord


def compute_source_id(path: Path) -> str:
    """由文件绝对路径生成稳定来源 ID（SHA256 前 16 位十六进制）。"""
    resolved = str(path.resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


@dataclass
class IndexResult:
    """单文件索引结果统计。"""

    source_id: str
    filename: str
    chunk_count: int


@dataclass
class FileIndexReport:
    """目录索引中单个文件的处理结果。"""

    path: str
    status: str  # ok / error / skipped
    chunk_count: int = 0
    error: str = ""


@dataclass
class DirectoryIndexReport:
    """目录批量索引汇总报告。"""

    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    total_chunks: int = 0
    files: list[FileIndexReport] = field(default_factory=list)


class IndexingPipeline:
    """文档索引流水线（同步实现，MCP 工具层异步调用）。"""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingClient,
        store: ChromaStore,
    ) -> None:
        """构造流水线，依赖注入便于测试替换。"""
        self._settings = settings
        self._embedder = embedder
        self._store = store

    def index_document(self, path: Path) -> IndexResult:
        """索引单个文件（幂等：重复索引自动覆盖旧数据）。

        流程：load → chunk → embed → 先删旧 source_id → upsert。

        Raises:
            LoaderError: 文件不存在、格式不支持或解析失败。
            EmbeddingError: 向量化失败。
            VectorStoreError: 向量写入失败。
        """
        if not path.is_file():
            raise LoaderError(f"文件不存在: {path}")

        source_id = compute_source_id(path)
        document = load_document(path)
        chunks = chunk_text(
            document.text,
            size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
            base_metadata=document.metadata,
        )
        indexed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not chunks:
            # 文件内容为空（或仅空白）：清掉旧索引（若曾索引过），返回 0
            self._store.delete_source(source_id)
            return IndexResult(
                source_id=source_id, filename=path.name, chunk_count=0
            )

        vectors = self._embedder.embed_documents([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise EmbeddingError(f"向量数量不匹配: {len(vectors)} vs {len(chunks)}")
        records = [
            VectorRecord(
                id=f"{source_id}_{c.chunk_index}",
                text=c.text,
                vector=vectors[i],
                metadata={
                    **c.metadata,
                    "source_id": source_id,
                    "chunk_index": c.chunk_index,
                    "indexed_at": indexed_at,
                },
            )
            for i, c in enumerate(chunks)
        ]

        # 幂等：先删旧数据（来源不存在时删除 0 条，无害）
        self._store.delete_source(source_id)
        self._store.upsert_chunks(records)
        return IndexResult(
            source_id=source_id, filename=path.name, chunk_count=len(chunks)
        )

    def index_directory(
        self, dir_path: Path, recursive: bool = True
    ) -> DirectoryIndexReport:
        """批量索引目录下全部支持格式的文件，逐文件容错。

        单文件失败（解析/向量化/写入）只记录到报告，不中断整体流程。
        """
        if not dir_path.is_dir():
            raise LoaderError(f"目录不存在: {dir_path}")

        pattern = "**/*" if recursive else "*"
        report = DirectoryIndexReport()
        for file_path in sorted(dir_path.glob(pattern)):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                report.skipped += 1
                report.files.append(
                    FileIndexReport(path=str(file_path), status="skipped")
                )
                continue
            try:
                result = self.index_document(file_path)
            except Exception as exc:  # 容错：任何单文件异常不中断批量
                report.failed += 1
                report.files.append(
                    FileIndexReport(
                        path=str(file_path), status="error", error=str(exc)
                    )
                )
                continue
            report.indexed += 1
            report.total_chunks += result.chunk_count
            report.files.append(
                FileIndexReport(
                    path=str(file_path),
                    status="ok",
                    chunk_count=result.chunk_count,
                )
            )
        return report

    def delete_source(self, source_id: str) -> int:
        """删除指定来源的全部向量，返回删除条数（0 表示来源不存在）。"""
        return self._store.delete_source(source_id)
