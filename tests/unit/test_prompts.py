"""提示词模板测试：引用格式统一为 [n](文件名)。"""

from __future__ import annotations

from rag_mcp_server.prompts.rag import (
    RAG_SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)
from rag_mcp_server.retriever.retriever import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(text="片段一", filename="a.txt", source="/x/a.txt", score=0.9, rank=1),
        RetrievedChunk(text="片段二", filename="b.md", source="/x/b.md", score=0.8, rank=2),
    ]


def test_context_block_format() -> None:
    block = build_context_block(_chunks())
    assert "[1](a.txt)\n片段一" in block
    assert "[2](b.md)\n片段二" in block


def test_user_prompt_contains_context_and_question() -> None:
    prompt = build_user_prompt("什么是苹果？", _chunks())
    assert prompt.startswith("参考资料：")
    assert "[1](a.txt)" in prompt
    assert "问题：什么是苹果？" in prompt


def test_system_prompt_requires_citation_format() -> None:
    assert "[n](文件名)" in RAG_SYSTEM_PROMPT
