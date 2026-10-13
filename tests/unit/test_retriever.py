"""检索器测试：空查询、精确匹配排序、top_k、rerank 钩子。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_mcp_server.common.exceptions import RetrievalError
from rag_mcp_server.config.settings import Settings
from rag_mcp_server.indexing.pipeline import IndexingPipeline
from rag_mcp_server.retriever.retriever import Retriever
from rag_mcp_server.tools.context import ToolContext, create_tool_context


def _make_ctx(ctx: ToolContext, tmp_path: Path, **overrides: object) -> ToolContext:
    """基于 ctx 的 embedder 构建自定义配置的上下文（小 chunk_size 便于分块）。"""
    base: dict[str, object] = dict(
        embedding_api_key="",
        llm_api_key="",
        embedding_dim=64,
        chroma_persist_dir=str(tmp_path / "chroma_custom"),
        log_dir=str(tmp_path / "logs_custom"),
    )
    base.update(overrides)
    settings = Settings(**base)  # type: ignore[arg-type]
    new_ctx = create_tool_context(settings)
    assert ctx.embedder is not None
    new_ctx.embedder = ctx.embedder
    return new_ctx


def _index_two_chunks(ctx: ToolContext, tmp_path: Path) -> str:
    """索引含两个独立段落（两个 chunk）的文件，返回第一个段落原文。"""
    p1 = "苹果是一种常见的水果，富含多种维生素。"  # 19 字符
    p2 = "香蕉富含钾元素。"  # 8 字符
    f = tmp_path / "fruits.txt"
    f.write_text(p1 + "\n\n" + p2, encoding="utf-8")
    IndexingPipeline(ctx.settings, ctx.embedder, ctx.store).index_document(f)
    return p1


def test_retrieve_empty_query_raises(ctx: ToolContext) -> None:
    with pytest.raises(RetrievalError, match="查询不能为空"):
        Retriever(ctx.settings, ctx.embedder, ctx.store).retrieve("   ")


def test_retrieve_empty_store_returns_empty(ctx: ToolContext) -> None:
    retriever = Retriever(ctx.settings, ctx.embedder, ctx.store)
    assert retriever.retrieve("任意查询") == []


def test_retrieve_ranks_exact_match_first(ctx: ToolContext, tmp_path: Path) -> None:
    custom = _make_ctx(ctx, tmp_path, chunk_size=25, chunk_overlap=0)
    p1 = _index_two_chunks(custom, tmp_path)

    results = Retriever(custom.settings, custom.embedder, custom.store).retrieve(p1)
    assert len(results) == 2
    # 哈希向量下完整块文本查询相似度≈1，必排第一
    assert results[0].text == p1
    assert results[0].rank == 1
    assert results[0].filename == "fruits.txt"
    assert results[0].source == str((tmp_path / "fruits.txt").resolve())
    assert results[0].score >= results[1].score


def test_retrieve_respects_top_k(ctx: ToolContext, tmp_path: Path) -> None:
    custom = _make_ctx(ctx, tmp_path, chunk_size=25, chunk_overlap=0)
    p1 = _index_two_chunks(custom, tmp_path)
    retriever = Retriever(custom.settings, custom.embedder, custom.store)
    assert len(retriever.retrieve(p1, top_k=1)) == 1


def test_rerank_hook_passthrough(ctx: ToolContext, tmp_path: Path) -> None:
    """rerank_enabled=True 时预留钩子为直通，不改变结果。"""
    custom = _make_ctx(ctx, tmp_path, chunk_size=25, chunk_overlap=0, rerank_enabled=True)
    p1 = _index_two_chunks(custom, tmp_path)
    results = Retriever(custom.settings, custom.embedder, custom.store).retrieve(p1)
    assert results[0].rank == 1
    assert results[0].text == p1
