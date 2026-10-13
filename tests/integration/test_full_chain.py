"""端到端集成：索引 → 列表 → 检索 → 问答降级 → 幂等重索引 → 删除。

使用 FakeEmbedder（不出网），覆盖 MCP 工具层 impl 的完整协作链路。
"""

from __future__ import annotations

from pathlib import Path

from rag_mcp_server.config.settings import Settings
from rag_mcp_server.tools.context import ToolContext, create_tool_context
from rag_mcp_server.tools.document_tools import (
    add_document_impl,
    delete_source_impl,
    list_sources_impl,
)
from rag_mcp_server.tools.search_tools import ask_knowledge_impl, search_knowledge_impl


async def _make_ctx(tmp_path: Path) -> ToolContext:
    from conftest import DIM, FakeEmbedder

    settings = Settings(
        embedding_api_key="",
        llm_api_key="",
        embedding_dim=DIM,
        chunk_size=50,
        chunk_overlap=0,
        chroma_persist_dir=str(tmp_path / "chroma"),
        log_dir=str(tmp_path / "logs"),
    )
    context = create_tool_context(settings)
    context.embedder = FakeEmbedder()
    return context


async def test_full_lifecycle(tmp_path: Path) -> None:
    ctx = await _make_ctx(tmp_path)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "apple.txt").write_text(
        "苹果是一种常见的水果，富含多种维生素。", encoding="utf-8"
    )
    (docs / "note.md").write_text("香蕉富含钾元素，是一种热带水果。", encoding="utf-8")
    (docs / "broken.docx").write_bytes(b"broken")  # 解析失败
    (docs / "ignore.xyz").write_text("?", encoding="utf-8")  # 不支持格式

    # 1) 目录批量索引：容错（1 成功 1 失败 1 跳过）
    report = await add_document_impl(ctx, str(docs), recursive=False)
    assert report["ok"] is True
    assert report["indexed"] == 2
    assert report["failed"] == 1
    assert report["skipped"] == 1
    assert report["total_chunks"] == 2

    # 2) 列表
    listing = await list_sources_impl(ctx)
    assert listing["total"] == 2
    assert {s["filename"] for s in listing["sources"]} == {"apple.txt", "note.md"}

    # 3) 检索：完整块文本查询 → 精确命中
    s = await search_knowledge_impl(
        ctx, "苹果是一种常见的水果，富含多种维生素。", top_k=2
    )
    assert s["count"] == 2
    assert s["results"][0]["rank"] == 1
    assert s["results"][0]["filename"] == "apple.txt"

    # 4) 问答：无 LLM Key → 降级返回片段
    a = await ask_knowledge_impl(ctx, "苹果是什么？")
    assert a["ok"] is True
    assert a["degraded"] is True
    assert "未配置 LLM_API_KEY" in a["answer"]
    assert {ref["filename"] for ref in a["references"]} == {"apple.txt", "note.md"}

    # 5) 幂等重索引：入库总量不变
    before = (await list_sources_impl(ctx))["sources"]
    await add_document_impl(ctx, str(docs / "apple.txt"))
    after = (await list_sources_impl(ctx))["sources"]
    assert sum(s["chunk_count"] for s in before) == sum(
        s["chunk_count"] for s in after
    )

    # 6) 删除单个来源
    target = next(s for s in after if s["filename"] == "apple.txt")
    deleted = await delete_source_impl(ctx, target["source_id"])
    assert deleted["ok"] is True
    assert deleted["deleted"] == 1
    remaining = (await list_sources_impl(ctx))["total"]
    assert remaining == 1
