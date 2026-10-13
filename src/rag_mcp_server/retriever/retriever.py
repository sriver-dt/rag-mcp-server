"""检索器：查询向量化 → Chroma 相似度检索 → 结构化结果。

当前不含重排序；RERANK_ENABLED=true 时调用预留的 _rerank 钩子
（默认关闭且为直通实现，接入 rerank 模型后补充逻辑）。
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_mcp_server.common.exceptions import RetrievalError
from rag_mcp_server.config.settings import Settings
from rag_mcp_server.embeddings.base import EmbeddingClient
from rag_mcp_server.vectorstore.chroma_store import ChromaStore


@dataclass
class RetrievedChunk:
    """单条检索结果（rank 从 1 开始，score 为相似度 [-1, 1]）。"""

    text: str
    filename: str
    source: str  # 文件原始路径
    score: float
    rank: int


class Retriever:
    """知识库检索器（同步实现，MCP 工具层异步调用）。"""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingClient,
        store: ChromaStore,
    ) -> None:
        """构造检索器，依赖注入便于测试替换。"""
        self._settings = settings
        self._embedder = embedder
        self._store = store

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """检索与 query 最相关的知识片段。

        Args:
            query: 查询文本，不能为空。
            top_k: 返回条数，缺省取配置 RETRIEVAL_TOP_K。

        Raises:
            RetrievalError: 查询为空。
            EmbeddingError: 查询向量化失败。
            VectorStoreError: 相似度检索失败。
        """
        if not query.strip():
            raise RetrievalError("查询不能为空")
        k = top_k if top_k and top_k > 0 else self._settings.retrieval_top_k

        vector = self._embedder.embed_query(query)
        hits = self._store.query(vector, top_k=k)
        chunks = [
            RetrievedChunk(
                text=h.text,
                filename=str(h.metadata.get("filename", "")),
                source=str(h.metadata.get("source", "")),
                score=round(h.similarity, 4),
                rank=i + 1,
            )
            for i, h in enumerate(hits)
        ]
        if self._settings.rerank_enabled:
            chunks = self._rerank(query, chunks)
        return chunks

    def _rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """重排序钩子（预留接口位，RERANK_ENABLED 默认关闭）。

        TODO: 接入 rerank 模型（如 bge-reranker API）后实现：
        按 query 与片段相关性重算分数并重排 rank。
        """
        _ = query  # 预留参数，接入模型后使用
        return chunks
