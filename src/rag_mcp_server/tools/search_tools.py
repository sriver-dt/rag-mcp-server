"""检索问答类 MCP 工具：search_knowledge / ask_knowledge。

ask_knowledge 的降级策略（任一 LLM 环节失败都优雅回退，不抛错中断）：
- 知识库无相关内容 → 直接提示，不调用 LLM
- 未配置 LLM_API_KEY / openai 未安装 / LLM 调用失败 → 降级返回检索片段原文
- LLM 正常 → 检索片段组装 prompt 生成带引用的回答

同步阻塞说明：embed 检索、Chroma 查询、LLM HTTP 调用均为阻塞操作，
统一经 asyncio.to_thread 移出事件循环线程执行。
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import MCPServer

from rag_mcp_server.config.settings import Settings
from rag_mcp_server.prompts.rag import RAG_SYSTEM_PROMPT, build_user_prompt
from rag_mcp_server.retriever.retriever import RetrievedChunk
from rag_mcp_server.tools.context import ToolContext, tool_safe

if TYPE_CHECKING:
    from openai import OpenAI


def _references(results: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """提取引用列表（不含原文，保持回答结果紧凑）。"""
    return [
        {"rank": r.rank, "filename": r.filename, "score": r.score}
        for r in results
    ]


def _format_fallback_answer(results: list[RetrievedChunk]) -> str:
    """降级回答：按引用格式罗列检索片段原文。

    与 prompts.build_context_block 的差异是有意设计：本函数面向
    人类阅读（附带 score 便于判断相关性），而 prompt 上下文面向
    LLM（score 反而可能干扰生成）。
    """
    lines = [
        f"[{r.rank}]({r.filename}) score={r.score}\n{r.text}"
        for r in results
    ]
    return "\n\n".join(lines)


def _chat_completion(
    client: OpenAI,
    settings: Settings,
    question: str,
    results: list[RetrievedChunk],
) -> str:
    """调用 OpenAI 兼容 LLM 生成回答（阻塞调用，需在线程池中执行）。

    client 由 ToolContext 缓存复用，避免每次请求重建连接池。
    """
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, results)},
        ],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


# ---------- 业务实现（可独立测试） ----------


async def search_knowledge_impl(
    ctx: ToolContext, query: str, top_k: int | None = None
) -> dict[str, Any]:
    """纯检索：返回与 query 最相关的知识片段（不调用 LLM）。"""
    # retrieve 内含 embedding API 调用与 Chroma 磁盘查询，均为阻塞操作
    results = await asyncio.to_thread(
        ctx.get_retriever().retrieve, query, top_k
    )
    return {
        "ok": True,
        "query": query,
        "count": len(results),
        "results": [asdict(r) for r in results],
    }


async def ask_knowledge_impl(
    ctx: ToolContext, question: str, top_k: int | None = None
) -> dict[str, Any]:
    """RAG 问答：检索 → 组装提示词 → LLM 生成（带降级路径）。"""
    results = await asyncio.to_thread(
        ctx.get_retriever().retrieve, question, top_k
    )

    if not results:
        return {
            "ok": True,
            "answer": (
                "知识库中没有找到与问题相关的内容。"
                "可先通过 add_document 工具添加相关文档后再提问。"
            ),
            "degraded": True,
            "references": [],
        }

    if not ctx.settings.llm_configured:
        # answer 为人类可读降级输出；references 供客户端结构化消费，
        # 二者内容重叠是有意设计（不同消费方），保留。
        return {
            "ok": True,
            "degraded": True,
            "answer": (
                "未配置 LLM_API_KEY，无法生成归纳回答，"
                "以下为检索到的最相关片段：\n\n"
                + _format_fallback_answer(results)
            ),
            "references": _references(results),
        }

    try:
        client = ctx.get_llm_client()
        answer = await asyncio.to_thread(
            _chat_completion, client, ctx.settings, question, results
        )
    except Exception as exc:
        # LLM 不可用（client 构建 / 网络 / API 失败）→ 降级为片段原文
        return {
            "ok": True,
            "degraded": True,
            "answer": (
                f"LLM 调用失败（{exc}），以下为检索到的最相关片段：\n\n"
                + _format_fallback_answer(results)
            ),
            "references": _references(results),
        }

    return {
        "ok": True,
        "degraded": False,
        "answer": answer,
        "references": _references(results),
    }


# ---------- MCP 工具注册 ----------


def register_search_tools(server: MCPServer, ctx: ToolContext) -> None:
    """向 MCPServer 注册检索问答类工具。"""

    @server.tool()
    async def search_knowledge(query: str, top_k: int | None = None) -> str:
        """在知识库中检索与查询最相关的原文片段（不调用 LLM）。

        适用于：查看知识库中有哪些相关内容、获取原文出处。

        Args:
            query: 查询文本。
            top_k: 返回的片段数量上限，缺省使用服务配置的 RETRIEVAL_TOP_K。
        """
        return await tool_safe(search_knowledge_impl(ctx, query, top_k))

    @server.tool()
    async def ask_knowledge(question: str, top_k: int | None = None) -> str:
        """基于知识库回答问题（RAG 问答）。

        先检索相关片段，再由 LLM 依据片段生成带引用 [n](文件名) 的回答；
        未配置 LLM 或调用失败时降级返回检索片段原文。

        Args:
            question: 用户问题。
            top_k: 参与回答的片段数量上限，缺省使用服务配置。
        """
        return await tool_safe(ask_knowledge_impl(ctx, question, top_k))


__all__ = [
    "ask_knowledge_impl",
    "register_search_tools",
    "search_knowledge_impl",
]
