"""Chroma 向量库封装。

职责边界：
- 只负责向量的持久化读写与相似度检索，不关心文本如何切分、向量从哪来
- 数据模型（VectorRecord/SearchHit/SourceInfo）为纯数据类，便于上层组装

设计要点：
- PersistentClient 本地持久化，目录来自配置 CHROMA_PERSIST_DIR
- 相似度度量固定 cosine，检索结果统一返回 similarity = 1 - distance
- 元数据 schema：source_id（路径哈希前 16 位）/ source / filename / 
  chunk_index / indexed_at，支持按 source_id 精确删除实现幂等索引
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_mcp_server.common.exceptions import VectorStoreError

# upsert 单批最大条数（Chroma 单请求的安全上限）
_UPSERT_BATCH_SIZE = 500


def _extract_layer(result: dict[str, Any], key: str) -> list[Any]:
    """从 Chroma 单查询结果中取出指定键的第一层列表，空结果返回空列表。

    Chroma 的 query 返回形如 ``{"ids": [[...]], "documents": [[...]]}`` 的双层结构，
    本方法统一做空值归一，避免上层对 Optional 做繁琐判断。
    """
    value = result.get(key)
    if not value:
        return []
    first = value[0]
    return list(first) if first else []


@dataclass
class VectorRecord:
    """写入向量库的最小单元。"""

    id: str  # 唯一 ID（索引阶段由 source_id + chunk_index 生成）
    text: str  # 原文内容
    vector: list[float]  # 向量
    metadata: dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass
class SearchHit:
    """单条检索命中结果。"""

    id: str
    text: str
    metadata: dict[str, Any]
    # 相似度 = 1 - cosine distance，取值范围 [-1, 1]，越大越相关；
    # 不做截断，保留原始语义供上层做阈值过滤
    similarity: float


@dataclass
class SourceInfo:
    """已索引文档来源的统计信息。"""

    source_id: str
    source: str  # 文件原始路径（或标识）
    filename: str
    chunk_count: int
    indexed_at: str  # 最近一次索引时间


class ChromaStore:
    """Chroma 持久化客户端的轻量封装。"""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        expected_dim: int | None = None,
    ) -> None:
        """初始化并校验向量维度。

        Args:
            persist_dir: 持久化目录，不存在时自动创建。
            collection_name: 集合名称。
            expected_dim: 期望向量维度（EMBEDDING_DIM）。集合非空时校验，
                不一致立即报错，避免脏数据入库。

        Raises:
            VectorStoreError: 客户端创建失败或维度不匹配。
        """
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            # chromadb 的官方类型标注对 ndarray 联合类型过于严苛，
            # 集合对象按 Any 使用，错误处理由本类各方法自行包装
            self._collection: Any = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Chroma 客户端初始化失败（dir={persist_dir}, "
                f"collection={collection_name}）: {exc}"
            ) from exc

        self._validate_dim(expected_dim)

    # ---------- 写入 / 删除 ----------

    def upsert_chunks(self, records: list[VectorRecord]) -> int:
        """批量写入（幂等：相同 ID 覆盖旧值），返回写入条数。"""
        if not records:
            return 0
        for start in range(0, len(records), _UPSERT_BATCH_SIZE):
            batch = records[start : start + _UPSERT_BATCH_SIZE]
            try:
                self._collection.upsert(
                    ids=[r.id for r in batch],
                    embeddings=[r.vector for r in batch],
                    documents=[r.text for r in batch],
                    metadatas=[r.metadata for r in batch],
                )
            except Exception as exc:
                raise VectorStoreError(f"向量写入失败（批次 {start}）: {exc}") from exc
        return len(records)

    def delete_source(self, source_id: str) -> int:
        """删除指定来源的全部向量，返回删除前的条数（0 表示来源不存在）。"""
        count = self._count_by_source(source_id)
        if count == 0:
            return 0
        try:
            self._collection.delete(where={"source_id": source_id})
        except Exception as exc:
            raise VectorStoreError(
                f"删除来源失败（source_id={source_id}）: {exc}"
            ) from exc
        return count

    # ---------- 检索 / 统计 ----------

    def query(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """按向量相似度检索，返回按相似度降序排列的命中列表。

        n_results 取 min(top_k, 可用条数)：无过滤时可用条数为全表总数；
        带 where 过滤时先统计过滤后的条数，避免请求超出可用数导致报错。
        """
        if top_k <= 0:
            return []
        total = self.count()
        if total == 0:
            return []
        limit = min(top_k, total)
        if where is not None:
            filtered = self._filtered_count(where)
            if filtered == 0:
                return []
            limit = min(top_k, filtered)
        try:
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"向量检索失败: {exc}") from exc

        hits: list[SearchHit] = []
        ids = _extract_layer(result, "ids")
        documents = _extract_layer(result, "documents")
        metadatas = _extract_layer(result, "metadatas")
        distances = _extract_layer(result, "distances")
        for i, rid in enumerate(ids):
            similarity = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            hits.append(
                SearchHit(
                    id=rid,
                    text=documents[i] if i < len(documents) else "",
                    metadata=dict(metadatas[i]) if i < len(metadatas) else {},
                    similarity=similarity,
                )
            )
        return hits

    def list_sources(self) -> list[SourceInfo]:
        """列出所有已索引来源及其分块统计（按文件名排序）。"""
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"读取来源列表失败: {exc}") from exc

        grouped: dict[str, SourceInfo] = {}
        for meta in result.get("metadatas") or []:
            if not meta or "source_id" not in meta:
                continue
            sid = str(meta["source_id"])
            if sid in grouped:
                grouped[sid].chunk_count += 1
                # 保留最近的索引时间
                if str(meta.get("indexed_at", "")) > grouped[sid].indexed_at:
                    grouped[sid].indexed_at = str(meta["indexed_at"])
            else:
                grouped[sid] = SourceInfo(
                    source_id=sid,
                    source=str(meta.get("source", "")),
                    filename=str(meta.get("filename", "")),
                    chunk_count=1,
                    indexed_at=str(meta.get("indexed_at", "")),
                )
        return sorted(grouped.values(), key=lambda s: s.filename)

    def count(self) -> int:
        """当前集合中的向量总数。"""
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(f"读取向量总数失败: {exc}") from exc

    # ---------- 内部方法 ----------

    def _filtered_count(self, where: dict[str, Any]) -> int:
        """统计满足过滤条件的向量条数（用于限定检索 n_results 上限）。"""
        try:
            result = self._collection.get(where=where, include=["metadatas"])
            return len(result.get("ids") or [])
        except Exception as exc:
            raise VectorStoreError(f"统计过滤后条数失败: {exc}") from exc

    def _count_by_source(self, source_id: str) -> int:
        """统计指定来源的向量条数。"""
        try:
            result = self._collection.get(where={"source_id": source_id})
            return len(result.get("ids") or [])
        except Exception as exc:
            raise VectorStoreError(
                f"查询来源条数失败（source_id={source_id}）: {exc}"
            ) from exc

    def _validate_dim(self, expected_dim: int | None) -> None:
        """集合非空时校验已存向量维度与配置一致。"""
        if expected_dim is None or self._collection.count() == 0:
            return
        try:
            peek = self._collection.peek(limit=1)
        except Exception as exc:
            raise VectorStoreError(f"读取集合样本失败: {exc}") from exc
        embeddings = peek.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            stored_dim = len(embeddings[0])
            if stored_dim != expected_dim:
                raise VectorStoreError(
                    f"向量维度冲突: 集合中已存在 {stored_dim} 维向量，"
                    f"配置 EMBEDDING_DIM={expected_dim}。"
                    "更换模型后需先清空集合（删除持久化目录）"
                )
