"""ToolContext 与 tool_safe 测试：惰性构建、异常转换、LLM client 单例。"""

from __future__ import annotations

import json

import pytest

from rag_mcp_server.common.exceptions import EmbeddingError
from rag_mcp_server.config.settings import Settings
from rag_mcp_server.tools.context import ToolContext, tool_result_json, tool_safe


async def test_tool_safe_ok() -> None:
    async def ok() -> dict[str, object]:
        return {"ok": True, "msg": "中文"}

    out = await tool_safe(ok())
    assert json.loads(out) == {"ok": True, "msg": "中文"}
    assert "中文" in out  # ensure_ascii=False，中文原样输出


async def test_tool_safe_converts_rag_error() -> None:
    async def bad() -> dict[str, object]:
        raise EmbeddingError("缺 Key")

    assert json.loads(await tool_safe(bad())) == {"ok": False, "error": "缺 Key"}


async def test_tool_safe_converts_unknown_error() -> None:
    async def boom() -> dict[str, object]:
        raise RuntimeError("boom")

    data = json.loads(await tool_safe(boom()))
    assert data["ok"] is False
    assert "内部错误" in data["error"]


def test_tool_result_json_chinese_raw() -> None:
    assert "值" in tool_result_json({"k": "值"})


async def test_get_embedder_without_key_raises(settings: Settings) -> None:
    from rag_mcp_server.tools.context import create_tool_context

    context = create_tool_context(settings)
    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        context.get_embedder()
    assert context.embedder is None  # 构造失败不残留半成品


async def test_lazy_singletons(ctx: ToolContext) -> None:
    assert ctx.get_pipeline() is ctx.get_pipeline()
    assert ctx.get_retriever() is ctx.get_retriever()


async def test_llm_client_singleton(ctx: ToolContext) -> None:
    ctx.settings.llm_api_key = "test-key"  # openai SDK 拒绝空 Key
    assert ctx.get_llm_client() is ctx.get_llm_client()
