"""分块策略测试：段落合并、中英文句切、小数保护、硬切尾片、overlap 边界。"""

from __future__ import annotations

import pytest

from rag_mcp_server.chunking import chunk_text


def test_empty_text_returns_empty() -> None:
    assert chunk_text("   \n\t ", 100, 10) == []


def test_short_text_single_chunk_with_metadata() -> None:
    chunks = chunk_text("只有一个短句。", 100, 10, {"filename": "a.txt"})
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "只有一个短句。"
    assert chunks[0].metadata == {"filename": "a.txt"}


def test_chinese_sentence_split_without_space() -> None:
    """回归：中文句号后无空格也必须切开。"""
    text = "第一句。" * 30  # 120 字符 > size=50
    chunks = chunk_text(text, 50, 0)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 50 for c in chunks)
    assert all(c.text.endswith("。") for c in chunks)  # 不截断句子


def test_decimal_protected_from_split() -> None:
    """英文小数点后无空白不切（保护 1.5 / v2.0）。"""
    text = "the price is 1.5 today. " * 10
    chunks = chunk_text(text, 40, 0)
    assert len(chunks) >= 2
    assert all("1.5" in c.text for c in chunks)  # 小数完整保留


def test_hard_split_no_character_loss() -> None:
    text = "a" * 120
    chunks = chunk_text(text, 50, 0)
    assert [len(c.text) for c in chunks] == [50, 50, 20]
    assert "".join(c.text for c in chunks) == text


def test_hard_split_tail_merged_with_next_sentences() -> None:
    """回归：硬切尾片不孤立成过小 chunk，而是与后续句子合并。"""
    long_sent = "无标点超长句" * 30  # 180 字符 > size=50
    text = long_sent + "短句一。短句二。"
    chunks = chunk_text(text, 50, 0)
    last = chunks[-1].text
    assert last.endswith("短句二。")
    assert "短句一" in last  # 尾片与后续句子在同一块
    assert all(len(c.text) <= 50 for c in chunks)


def test_paragraphs_merged_with_blank_line() -> None:
    text = "第一段。\n\n第二段。\n\n第三段。"
    chunks = chunk_text(text, 100, 0)
    assert len(chunks) == 1
    assert chunks[0].text == text  # 段落间保留空行分隔


def test_overlap_repeats_prev_tail_without_cascade() -> None:
    text = "段落一内容。" * 20 + "\n\n" + "段落二内容。" * 20  # 两个超长段落
    chunks = chunk_text(text, 50, 10)
    assert len(chunks) >= 2
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        # 每块头部重复上一块（原始块）尾部 10 字符，且不级联滚雪球
        assert cur.text[:10] == prev.text[-10:]
        assert len(cur.text) <= 50 + 10  # 容忍 size+overlap


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("x", 10, -1)
    with pytest.raises(ValueError):
        chunk_text("x", 10, 10)
