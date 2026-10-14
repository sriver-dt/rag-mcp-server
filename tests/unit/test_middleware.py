"""客户端连接日志中间件的单元测试。"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from rag_mcp_server.server.middleware import (
    ConnectionLogMiddleware,
    _extract_client_info,
)


# caplog 只监听 root，项目 logger propagate=False 时需临时打开
# （monkeypatch 自动在用例结束后恢复原值）
@pytest.fixture(autouse=True)
def _propagate(monkeypatch):
    monkeypatch.setattr(logging.getLogger("rag_mcp_server"), "propagate", True)


# ---------- _extract_client_info ----------


def test_extract_full_client_info() -> None:
    client, version = _extract_client_info(
        {
            "protocolVersion": "2025-06-18",
            "clientInfo": {"name": "Trae IDE", "version": "1.2.0"},
        }
    )
    assert client == "Trae IDE v1.2.0"
    assert version == "2025-06-18"


def test_extract_missing_client_version() -> None:
    client, _ = _extract_client_info({"clientInfo": {"name": "curl"}})
    assert client == "curl v?"


def test_extract_no_client_info() -> None:
    client, version = _extract_client_info({"protocolVersion": "2025-06-18"})
    assert client == "未知客户端"
    assert version == "2025-06-18"


def test_extract_params_none() -> None:
    assert _extract_client_info(None) == ("未知客户端", "未知")


def test_extract_malformed_client_info() -> None:
    assert _extract_client_info({"clientInfo": "bad"}) == ("未知客户端", "未知")
    assert _extract_client_info({"clientInfo": {"version": "1.0"}}) == (
        "未知客户端",
        "未知",
    )


# ---------- ConnectionLogMiddleware ----------


def _make_ctx(method: str, params: Any, request_id: Any) -> SimpleNamespace:
    """构造中间件实际访问的最小 ctx（method/params/request_id）。"""
    return SimpleNamespace(method=method, params=params, request_id=request_id)


async def test_middleware_logs_initialize_and_passes_through(caplog) -> None:
    called: list[str] = []

    async def call_next(ctx: Any) -> dict[str, Any]:
        called.append(ctx.method)
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="rag_mcp_server"):
        result = await ConnectionLogMiddleware()(
            _make_ctx(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "Trae IDE", "version": "1.2.0"},
                },
                "req-1",
            ),
            call_next,
        )

    # 请求必须原样透传给后续链路
    assert result == {"ok": True}
    assert called == ["initialize"]
    assert "客户端已连接" in caplog.text
    assert "Trae IDE v1.2.0" in caplog.text


async def test_middleware_no_log_for_other_methods(caplog) -> None:
    async def call_next(ctx: Any) -> None:
        return None

    with caplog.at_level(logging.INFO, logger="rag_mcp_server"):
        await ConnectionLogMiddleware()(
            _make_ctx("tools/call", {"name": "list_sources"}, "req-2"), call_next
        )

    assert "客户端已连接" not in caplog.text


async def test_middleware_no_log_for_notification(caplog) -> None:
    """request_id 为 None 表示通知（initialize 不会以此形式出现），不记日志。"""
    async def call_next(ctx: Any) -> None:
        return None

    with caplog.at_level(logging.INFO, logger="rag_mcp_server"):
        await ConnectionLogMiddleware()(
            _make_ctx(
                "initialize",
                {"clientInfo": {"name": "x", "version": "1"}},
                None,
            ),
            call_next,
        )

    assert "客户端已连接" not in caplog.text
