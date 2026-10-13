"""共享测试夹具：FakeEmbedder（确定性哈希向量）+ 临时配置上下文。

不出网原则：所有测试用 FakeEmbedder 替代真实 embedding API；
LLM 客户端在需要时注入假对象（见 test_search_tools.py）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from rag_mcp_server.config.settings import Settings
from rag_mcp_server.embeddings.base import EmbeddingClient
from rag_mcp_server.tools.context import ToolContext, create_tool_context

# 测试向量维度（小维度加速 Chroma 本地计算，需与 Settings.embedding_dim 一致）
DIM = 64


class FakeEmbedder(EmbeddingClient):
    """确定性哈希向量（SHA256 归一化）。

    注意：哈希向量下只有"查询文本 == 完整块文本"时相似度才接近 1，
    需要断言排序的测试应使用完整块文本作为查询。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [b / 255.0 for b in digest[:DIM]]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """无任何 API Key 的基础配置（Chroma/日志指向临时目录）。"""
    return Settings(
        embedding_api_key="",
        llm_api_key="",
        embedding_dim=DIM,
        chroma_persist_dir=str(tmp_path / "chroma"),
        log_dir=str(tmp_path / "logs"),
    )


@pytest.fixture
def make_settings(tmp_path: Path):
    """Settings 工厂：允许单个测试覆盖任意字段（如 chunk_size）。"""

    def _make(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "embedding_api_key": "",
            "llm_api_key": "",
            "embedding_dim": DIM,
            "chroma_persist_dir": str(tmp_path / "chroma"),
            "log_dir": str(tmp_path / "logs"),
        }
        base.update(overrides)
        return Settings(**base)

    return _make


@pytest.fixture
def ctx(settings: Settings, fake_embedder: FakeEmbedder) -> ToolContext:
    """注入 FakeEmbedder 的工具上下文（可跑全链路，不出网）。"""
    context = create_tool_context(settings)
    context.embedder = fake_embedder
    return context
