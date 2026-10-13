"""Embedding 客户端抽象：定义统一的向量化接口。

阶段 1 提供协议定义与 OpenAI 兼容实现；后续如需本地模型，
只需新增实现该协议的类，上层（索引/检索）代码无需改动。
"""

from rag_mcp_server.embeddings.base import EmbeddingClient
from rag_mcp_server.embeddings.openai_compatible import OpenAICompatibleEmbedder

__all__ = ["EmbeddingClient", "OpenAICompatibleEmbedder"]
