"""Embedding 客户端协议。

约定：
- ``embed_documents`` 用于批量向量化（索引阶段），输入顺序与输出顺序一一对应
- ``embed_query`` 用于单条查询向量化（检索阶段）
- 实现类应保证返回向量维度与配置的 EMBEDDING_DIM 一致，不一致时抛出 EmbeddingError
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingClient(Protocol):
    """向量客户端协议（结构化类型，无需显式继承）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档片段，返回与输入等长且顺序对应的向量列表。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询文本。"""
        ...
