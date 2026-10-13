"""检索问答工具测试：检索、无 LLM 降级、LLM 故障降级、正常路径。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rag_mcp_server.config.settings import Settings
from rag_mcp_server.prompts.rag import RAG_SYSTEM_PROMPT
from rag_mcp_server.tools.context import ToolContext, create_tool_context
from rag_mcp_server.tools.document_tools import add_document_impl
from rag_mcp_server.tools.search_tools import ask_knowledge_impl, search_knowledge_impl


def _response(content: str) -> SimpleNamespace:
    """构造 openai ChatCompletion 形状的假响应。"""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class BrokenLLMClient:
    """模拟 LLM 网络故障的假客户端。"""

    class _Completions:
        def create(self, **kwargs: Any) -> object:
            raise RuntimeError("模拟 LLM 网络故障")

    class _Chat:
        def __init__(self) -> None:
            self.completions = BrokenLLMClient._Completions()

    def __init__(self) -> None:
        self.chat = BrokenLLMClient._Chat()


class FakeLLMClient:
    """记录请求参数并返回固定回答的假客户端。"""

    class _Completions:
        def __init__(self, answer: str, captured: dict[str, Any]) -> None:
            self._answer = answer
            self._captured = captured

        def create(self, **kwargs: Any) -> object:
            self._captured.update(kwargs)
            return _response(self._answer)

    class _Chat:
        def __init__(self, answer: str, captured: dict[str, Any]) -> None:
            self.completions = FakeLLMClient._Completions(answer, captured)

    def __init__(self, answer: str) -> None:
        self.captured: dict[str, Any] = {}
        self.chat = FakeLLMClient._Chat(answer, self.captured)


async def _indexed_ctx(tmp_path: Path, **overrides: object) -> ToolContext:
    """带 FakeEmbedder 与已索引文档的上下文（默认无 LLM Key）。"""
    base: dict[str, object] = dict(
        embedding_api_key="",
        llm_api_key="",
        embedding_dim=64,
        chroma_persist_dir=str(tmp_path / "chroma_search"),
        log_dir=str(tmp_path / "logs_search"),
    )
    base.update(overrides)
    settings = Settings(**base)  # type: ignore[arg-type]
    context = create_tool_context(settings)
    from conftest import FakeEmbedder

    context.embedder = FakeEmbedder()
    f = tmp_path / "kb.txt"
    f.write_text("苹果是一种常见的水果，富含多种维生素。", encoding="utf-8")
    await add_document_impl(context, str(f))
    return context


async def test_search_knowledge_returns_hits(tmp_path: Path) -> None:
    context = await _indexed_ctx(tmp_path)
    r = await search_knowledge_impl(
        context, "苹果是一种常见的水果，富含多种维生素。"
    )
    assert r["ok"] is True
    assert r["count"] == 1
    hit = r["results"][0]
    assert hit["rank"] == 1
    assert hit["filename"] == "kb.txt"
    assert isinstance(hit["score"], float)


async def test_search_knowledge_empty_store(tmp_path: Path) -> None:
    context = await _indexed_ctx(tmp_path)
    from rag_mcp_server.tools.document_tools import delete_source_impl, list_sources_impl

    for s in (await list_sources_impl(context))["sources"]:
        await delete_source_impl(context, s["source_id"])
    r = await search_knowledge_impl(context, "苹果")
    assert r["ok"] is True and r["count"] == 0


async def test_ask_knowledge_empty_results_prompts_add(
    tmp_path: Path,
) -> None:
    context = await _indexed_ctx(tmp_path)
    from rag_mcp_server.tools.document_tools import delete_source_impl, list_sources_impl

    for s in (await list_sources_impl(context))["sources"]:
        await delete_source_impl(context, s["source_id"])

    r = await ask_knowledge_impl(context, "什么是苹果？")
    assert r["ok"] is True
    assert r["degraded"] is True
    assert "没有找到" in r["answer"]
    assert r["references"] == []


async def test_ask_knowledge_degrades_without_llm_key(tmp_path: Path) -> None:
    context = await _indexed_ctx(tmp_path)  # llm_api_key=""
    r = await ask_knowledge_impl(context, "苹果是一种常见的水果，富含多种维生素。")
    assert r["ok"] is True
    assert r["degraded"] is True
    assert "未配置 LLM_API_KEY" in r["answer"]
    assert len(r["references"]) == 1
    assert r["references"][0]["filename"] == "kb.txt"


async def test_ask_knowledge_degrades_on_llm_failure(tmp_path: Path) -> None:
    context = await _indexed_ctx(tmp_path, llm_api_key="fake-key")
    context._llm_client = BrokenLLMClient()

    r = await ask_knowledge_impl(context, "苹果是一种常见的水果，富含多种维生素。")
    assert r["ok"] is True
    assert r["degraded"] is True
    assert "LLM 调用失败" in r["answer"]
    assert len(r["references"]) == 1


async def test_ask_knowledge_success_path(tmp_path: Path) -> None:
    context = await _indexed_ctx(tmp_path, llm_api_key="fake-key")
    fake_llm = FakeLLMClient("  答案 [1](kb.txt)  ")
    context._llm_client = fake_llm

    r = await ask_knowledge_impl(context, "苹果是什么？")
    assert r["ok"] is True
    assert r["degraded"] is False
    assert r["answer"] == "答案 [1](kb.txt)"  # strip 生效
    assert len(r["references"]) == 1

    # 请求参数校验：模型与消息结构
    assert fake_llm.captured["model"] == context.settings.llm_model
    messages = fake_llm.captured["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == RAG_SYSTEM_PROMPT
    assert "苹果是什么？" in messages[1]["content"]
    assert "[1](kb.txt)" in messages[1]["content"]
