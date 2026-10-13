"""文本分块：段落优先，超长段落再按句切。

目标：
- 尽量保留段落完整语义
- 相邻块保证 overlap（overlap 个字符由上一块尾部重复出现在下一块头部）
- 单块最大 size 字符（非 token，避免引入 tokenizer 依赖）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# 段落分隔符：空行、Markdown 标题、水平分割线、列表之间的空行
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
# 句子分隔符：中文句末标点后直接切（中文书写无空格习惯）；
# 英文句末标点后需跟空白才切，避免 "1.5"、"v2.0" 等小数/版本号误切
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")


@dataclass
class Chunk:
    """分块结果：纯文本 + 继承自原文档的元数据 + 索引。"""

    text: str
    chunk_index: int
    metadata: dict[str, Any]


def chunk_text(
    text: str,
    size: int,
    overlap: int,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """将文本切分为多个 Chunk。

    切分策略：
    1. 按段落边界切（空行/Markdown 标题）
    2. 短段落合并到同一 chunk，直到 size 上限
    3. 单个段落超过 size 时，按句子切分；句子仍超长时硬切
    4. 每个 chunk 尾部 overlap 个字符会在下一块头部重复

    Args:
        text: 原始纯文本。
        size: 单块最大字符数。
        overlap: 相邻块重叠字符数。
        base_metadata: 继承自原文档的元数据（source/filename 等）。

    Returns:
        有序 Chunk 列表，chunk_index 从 0 开始。
    """
    if not text.strip():
        return []
    if size <= 0:
        raise ValueError(f"size 必须为正整数，当前 {size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负，当前 {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap({overlap}) 必须小于 size({size})")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    raw_chunks: list[str] = _merge_paragraphs(paragraphs, size)

    # 对合并后仍超长的块做二次切分
    final_chunks: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) <= size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_long_chunk(chunk, size))

    # 应用 overlap
    overlapped = _apply_overlap(final_chunks, overlap)

    base = dict(base_metadata or {})
    return [
        Chunk(text=c, chunk_index=i, metadata=dict(base))
        for i, c in enumerate(overlapped)
        if c.strip()
    ]


def _merge_paragraphs(paragraphs: list[str], size: int) -> list[str]:
    """段落级合并：短段落拼到同一 chunk，直到触及 size 上限。

    每个 chunk 内段落间用空行分隔，保留语义边界。
    """
    merged: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    for para in paragraphs:
        if len(para) > size:
            # 超长段落：先把当前累积的段落收掉，再独立处理
            if current_lines:
                merged.append("\n\n".join(current_lines))
                current_lines, current_len = [], 0
            merged.append(para)  # 超长段留在列表里，后续 _split_long_chunk 处理
            continue
        if current_len == 0:
            current_lines.append(para)
            current_len = len(para)
        elif current_len + 2 + len(para) <= size:  # +2 是段落间的 \n\n
            current_lines.append(para)
            current_len += 2 + len(para)
        else:
            merged.append("\n\n".join(current_lines))
            current_lines = [para]
            current_len = len(para)
    if current_lines:
        merged.append("\n\n".join(current_lines))
    return merged


def _split_long_chunk(chunk: str, size: int) -> list[str]:
    """对单个超长段落做二次切分：先按句子，仍超长则硬切。"""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(chunk) if s.strip()]
    if not sentences:
        # 无句号分隔（如大段英文无标点），退化为硬切
        return _hard_split(chunk, size)

    result: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        if len(sent) > size:
            # 单句超长：先把当前缓冲收掉，再硬切此句；
            # 尾片留入缓冲与后续句子合并，避免产生过小的孤立块
            if buf:
                result.append(" ".join(buf))
                buf, buf_len = [], 0
            pieces = _hard_split(sent, size)
            result.extend(pieces[:-1])
            buf = [pieces[-1]]
            buf_len = len(pieces[-1])
            continue
        if buf_len == 0:
            buf.append(sent)
            buf_len = len(sent)
        elif buf_len + 1 + len(sent) <= size:  # 句子间空格
            buf.append(sent)
            buf_len += 1 + len(sent)
        else:
            result.append(" ".join(buf))
            buf = [sent]
            buf_len = len(sent)
    if buf:
        result.append(" ".join(buf))
    return result


def _hard_split(text: str, size: int) -> list[str]:
    """无标点/无段落边界时的兜底硬切（size 字符等长）。"""
    if len(text) <= size:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """在 chunk 间引入 overlap：每块头部重复上一块尾部的 overlap 个字符。

    首块不变；overlap 取自切分后的原始块（不级联，避免重叠滚雪球）。
    加入 overlap 后单块长度最多为 size + overlap（容忍值），
    不做截断——截断会丢失当前块尾部内容，违背 overlap 的召回目的。
    """
    if overlap == 0 or len(chunks) <= 1:
        return chunks

    result: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        prev_tail = prev[-overlap:] if len(prev) >= overlap else prev
        result.append(prev_tail + chunks[i])
    return result
