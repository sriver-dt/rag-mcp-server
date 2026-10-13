"""RAG 提示词模板：问答 system prompt 与上下文拼装。"""

from rag_mcp_server.prompts.rag import (
    RAG_SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)

__all__ = ["RAG_SYSTEM_PROMPT", "build_context_block", "build_user_prompt"]
