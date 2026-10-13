"""文档管理工具测试：无 Key 语义回归 + FakeEmbedder 全链路。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_mcp_server.common.exceptions import EmbeddingError, LoaderError
from rag_mcp_server.config.settings import Settings
from rag_mcp_server.tools.context import ToolContext, create_tool_context
from rag_mcp_server.tools.document_tools import (
    add_document_impl,
    delete_source_impl,
    list_sources_impl,
)


async def test_list_sources_empty(ctx: ToolContext) -> None:
    r = await list_sources_impl(ctx)
    assert r["ok"] is True
    assert r["total"] == 0


async def test_no_key_list_and_delete_usable(settings: Settings) -> None:
    """回归：无 EMBEDDING_API_KEY 时 list/delete 可用，且不构造 embedder/pipeline。"""
    context = create_tool_context(settings)

    r = await list_sources_impl(context)
    assert r["ok"] is True
    assert context.embedder is None

    r = await delete_source_impl(context, "nonexistent")
    assert r == {"ok": True, "source_id": "nonexistent", "deleted": 0}
    assert context.embedder is None  # 删除不应触发 embedder 构造
    assert context._pipeline is None  # 不应构建 pipeline


async def test_no_key_add_document_clear_error(
    settings: Settings, tmp_path: Path
) -> None:
    """无 Key 索引返回明确的配置错误（而非内部错误）。"""
    context = create_tool_context(settings)
    f = tmp_path / "a.txt"
    f.write_text("内容", encoding="utf-8")
    with pytest.raises(EmbeddingError, match="EMBEDDING_API_KEY"):
        await add_document_impl(context, str(f))


async def test_add_document_missing_path(ctx: ToolContext, tmp_path: Path) -> None:
    with pytest.raises(LoaderError, match="路径不存在"):
        await add_document_impl(ctx, str(tmp_path / "missing.txt"))


async def test_add_document_file_and_directory(
    ctx: ToolContext, tmp_path: Path
) -> None:
    f = tmp_path / "a.txt"
    f.write_text("甲" * 500 + "\n\n" + "乙" * 500, encoding="utf-8")
    r = await add_document_impl(ctx, str(f))
    assert r["ok"] is True
    assert r["mode"] == "file"
    assert r["chunk_count"] == 2

    d = tmp_path / "dir"
    (d / "sub").mkdir(parents=True)
    (d / "b.txt").write_text("丙" * 500 + "\n\n" + "丁" * 500, encoding="utf-8")
    (d / "sub" / "c.txt").write_text("戊" * 500, encoding="utf-8")
    (d / "skip.xyz").write_text("?", encoding="utf-8")

    r = await add_document_impl(ctx, str(d))  # recursive 默认 False
    assert r["mode"] == "directory"
    assert r["indexed"] == 1
    assert r["skipped"] == 1
    assert r["failed"] == 0

    r = await add_document_impl(ctx, str(d), recursive=True)
    assert r["indexed"] == 2
    assert r["total_chunks"] == 3


async def test_delete_real_data(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("甲" * 500 + "\n\n" + "乙" * 500, encoding="utf-8")
    r = await add_document_impl(ctx, str(f))
    source_id = r["source_id"]

    r = await delete_source_impl(ctx, source_id)
    assert r["ok"] is True
    assert r["deleted"] == 2
    assert (await list_sources_impl(ctx))["total"] == 0
