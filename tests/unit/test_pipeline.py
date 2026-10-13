"""索引流水线测试：单文件幂等、目录批量容错、source_id 稳定性。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_mcp_server.common.exceptions import LoaderError
from rag_mcp_server.indexing.pipeline import IndexingPipeline, compute_source_id
from rag_mcp_server.tools.context import ToolContext


def _pipe(ctx: ToolContext) -> IndexingPipeline:
    assert ctx.embedder is not None
    return IndexingPipeline(ctx.settings, ctx.embedder, ctx.store)


def test_source_id_stable_and_length(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    assert compute_source_id(f) == compute_source_id(f)
    assert len(compute_source_id(f)) == 16


def test_index_document_and_idempotent(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("甲" * 500 + "\n\n" + "乙" * 500, encoding="utf-8")
    pipe = _pipe(ctx)

    result = pipe.index_document(f)
    assert result.chunk_count == 2
    assert result.source_id == compute_source_id(f)
    assert ctx.store.count() == 2

    pipe.index_document(f)  # 幂等：重复索引覆盖旧数据
    assert ctx.store.count() == 2


def test_index_document_missing_file(ctx: ToolContext, tmp_path: Path) -> None:
    with pytest.raises(LoaderError, match="文件不存在"):
        _pipe(ctx).index_document(tmp_path / "missing.txt")


def test_index_empty_file(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("  \n\t ", encoding="utf-8")
    result = _pipe(ctx).index_document(f)
    assert result.chunk_count == 0
    assert ctx.store.count() == 0


def test_index_directory_mixed_report(ctx: ToolContext, tmp_path: Path) -> None:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "ok.txt").write_text("甲" * 500 + "\n\n" + "乙" * 500, encoding="utf-8")
    (d / "skip.xyz").write_text("?", encoding="utf-8")
    (d / "bad.docx").write_bytes(b"garbage")  # 解析失败，不中断批量

    report = _pipe(ctx).index_directory(d, recursive=False)
    assert (report.indexed, report.failed, report.skipped) == (1, 1, 1)
    assert report.total_chunks == 2
    by_status = {r.status: r for r in report.files}
    assert by_status["ok"].chunk_count == 2
    assert by_status["error"].error  # 失败原因已记录


def test_index_directory_recursive_flag(ctx: ToolContext, tmp_path: Path) -> None:
    base = tmp_path / "tree"
    (base / "sub").mkdir(parents=True)
    (base / "top.txt").write_text("顶层" * 300, encoding="utf-8")
    (base / "sub" / "inner.txt").write_text("内层" * 300, encoding="utf-8")
    pipe = _pipe(ctx)
    assert pipe.index_directory(base, recursive=True).indexed == 2
    assert pipe.index_directory(base, recursive=False).indexed == 1


def test_pipeline_delete_source(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("甲" * 500, encoding="utf-8")
    pipe = _pipe(ctx)
    source_id = pipe.index_document(f).source_id
    assert pipe.delete_source(source_id) == 1
    assert pipe.delete_source(source_id) == 0  # 再删返回 0
