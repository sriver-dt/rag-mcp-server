"""RAG 问答提示词模板。

约束 LLM 仅依据检索片段回答并标注引用，检索片段不足时明确告知，
避免幻觉。上下文块格式与引用标记 [n](文件名) 呼应。
"""

from __future__ import annotations

from rag_mcp_server.retriever.retriever import RetrievedChunk

RAG_SYSTEM_PROMPT = """你是一个严谨的中文知识库问答助手，必须遵守以下规则：

1. 仅依据下方"参考资料"回答问题，禁止编造资料中不存在的信息。
2. 回答中使用到的片段需在句末标注引用，格式为 [n](文件名)，n 为片段编号。
3. 当参考资料不足以回答问题时，明确说明"知识库中暂无足够信息"，
   并可建议用户先添加相关文档（add_document 工具）。
4. 回答保持简洁、结构清晰，使用中文。"""


def build_context_block(results: list[RetrievedChunk]) -> str:
    """将检索结果拼装为带编号的上下文块。

    格式：
        [1](文件名)
        片段原文
    """
    blocks = [f"[{r.rank}]({r.filename})\n{r.text}" for r in results]
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str, results: list[RetrievedChunk]
) -> str:
    """拼装最终 user 消息：参考资料 + 用户问题。"""
    context = build_context_block(results)
    return f"参考资料：\n\n{context}\n\n问题：{question}"
