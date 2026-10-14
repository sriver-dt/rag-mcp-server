"""OpenAI 兼容的 Embedding 实现。

支持任何兼容 ``/v1/embeddings`` 接口的服务（OpenAI、各厂商代理、
本地 vLLM/ollama 网关等），通过配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY /
EMBEDDING_MODEL / EMBEDDING_DIM / EMBEDDING_BATCH_SIZE 控制。

- 自动分批请求（每批条数由 EMBEDDING_BATCH_SIZE 控制，默认 20），
  避免超过服务端的单请求输入上限（如 DashScope 系列模型限 20 条）
- 严格校验返回向量数量与维度，异常统一包装为 EmbeddingError
"""

from __future__ import annotations

from openai import OpenAI

from rag_mcp_server.common.exceptions import EmbeddingError
from rag_mcp_server.config.settings import Settings


class OpenAICompatibleEmbedder:
    """基于 openai SDK 的 Embedding 客户端（同步调用）。"""

    def __init__(self, settings: Settings) -> None:
        """初始化客户端。

        Args:
            settings: 全局配置（读取 EMBEDDING_* 字段，单批条数取
                ``EMBEDDING_BATCH_SIZE``，已由聚合校验保证为正整数）。

        Raises:
            EmbeddingError: 未配置 API Key。
        """
        if not settings.embedding_api_key.strip():
            raise EmbeddingError(
                "EMBEDDING_API_KEY 未配置，无法使用 OpenAI 兼容向量化服务"
            )
        self._model = settings.embedding_model
        self._expected_dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        self._client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化：自动分批请求，返回与输入等长且顺序对应的向量列表。"""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._request(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询文本。"""
        if not text.strip():
            raise EmbeddingError("查询文本为空，无法向量化")
        return self._request([text])[0]

    def _request(self, batch: list[str]) -> list[list[float]]:
        """执行单次 embeddings API 请求并校验结果。"""
        if not batch:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            # 响应解析一并纳入 try：部分兼容网关异常时可能返回
            # data=None 等畸形结构，需统一包装为 EmbeddingError
            data = response.data
            if not data:
                raise EmbeddingError(
                    f"Embedding API 返回空数据（data=None/[]，model={self._model}）"
                )
            # 按返回的 index 排序，保证与输入顺序一一对应
            ordered = sorted(data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise EmbeddingError(
                    f"Embedding 返回数量不匹配: 请求 {len(batch)} 条，返回 {len(ordered)} 条"
                )
            vectors = [item.embedding for item in ordered]
        except EmbeddingError:
            raise
        except Exception as exc:  # openai 的异常体系庞大，统一包装
            raise EmbeddingError(
                f"Embedding API 请求失败（model={self._model}）: {exc}"
            ) from exc

        # 以下为纯内存校验（batch 非空且数量匹配，vectors 必然非空）
        actual_dim = len(vectors[0])
        if any(len(v) != actual_dim for v in vectors):
            raise EmbeddingError("Embedding 返回的向量维度不一致")
        if actual_dim != self._expected_dim:
            raise EmbeddingError(
                f"Embedding 维度不匹配: 模型返回 {actual_dim} 维，"
                f"配置 EMBEDDING_DIM={self._expected_dim}。"
                "请修正 EMBEDDING_DIM 或更换模型"
            )
        return vectors
