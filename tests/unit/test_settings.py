"""Settings 约束校验测试：负 overlap、top_k 上界、transport、错误聚合。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_mcp_server.common.exceptions import ConfigError
from rag_mcp_server.config.settings import Settings


def test_default_settings_valid(settings: Settings) -> None:
    assert settings.mcp_transport == "stdio"
    assert settings.retrieval_top_k == 5
    assert settings.embedding_configured is False
    assert settings.llm_configured is False


def test_negative_overlap_rejected() -> None:
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        Settings(chunk_overlap=-1, embedding_api_key="", llm_api_key="")


def test_overlap_ge_size_rejected() -> None:
    with pytest.raises(ConfigError, match="必须小于 CHUNK_SIZE"):
        Settings(
            chunk_size=100,
            chunk_overlap=100,
            embedding_api_key="",
            llm_api_key="",
        )


def test_top_k_bounds_rejected() -> None:
    with pytest.raises(ConfigError, match="正整数"):
        Settings(retrieval_top_k=0, embedding_api_key="", llm_api_key="")
    with pytest.raises(ConfigError, match="上限"):
        Settings(retrieval_top_k=51, embedding_api_key="", llm_api_key="")


def test_invalid_transport_rejected() -> None:
    with pytest.raises(ConfigError, match="MCP_TRANSPORT"):
        Settings(mcp_transport="websocket", embedding_api_key="", llm_api_key="")


def test_non_positive_dim_rejected() -> None:
    with pytest.raises(ConfigError, match="EMBEDDING_DIM"):
        Settings(embedding_dim=0, embedding_api_key="", llm_api_key="")


def test_non_positive_chunk_size_rejected() -> None:
    with pytest.raises(ConfigError, match="CHUNK_SIZE"):
        Settings(chunk_size=0, embedding_api_key="", llm_api_key="")


def test_errors_aggregated_in_one_raise() -> None:
    """多个配置错误应一次性聚合抛出，而非逐条中断。"""
    with pytest.raises(ConfigError) as exc_info:
        Settings(
            chunk_overlap=-1,
            mcp_transport="bogus",
            embedding_api_key="",
            llm_api_key="",
        )
    msg = str(exc_info.value)
    assert "CHUNK_OVERLAP" in msg
    assert "MCP_TRANSPORT" in msg


def test_configured_flags() -> None:
    s = Settings(embedding_api_key="ek", llm_api_key="lk", embedding_dim=64)
    assert s.embedding_configured is True
    assert s.llm_configured is True


def test_absolute_persist_dir_wins(tmp_path: Path) -> None:
    """绝对路径作为 CHROMA_PERSIST_DIR 时不应再拼接项目根。"""
    target = tmp_path / "db"
    s = Settings(
        chroma_persist_dir=str(target),
        embedding_api_key="",
        llm_api_key="",
    )
    assert s.chroma_persist_path == target.resolve()
