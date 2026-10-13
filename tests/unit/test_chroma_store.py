"""ChromaStore 单元测试（tmp_path 临时持久化目录）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_mcp_server.common.exceptions import VectorStoreError
from rag_mcp_server.vectorstore.chroma_store import ChromaStore, VectorRecord

DIM = 8


def make_store(
    tmp_path: Path, name: str = "test_collection", dim: int | None = None
) -> ChromaStore:
    return ChromaStore(tmp_path / "db", name, expected_dim=dim)


def rec(rid: str, text: str, source_id: str, chunk_index: int = 0) -> VectorRecord:
    """构造确定性测试向量记录。"""
    vector = [0.1 * ((i + chunk_index) % DIM + 1) for i in range(DIM)]
    return VectorRecord(
        id=rid,
        text=text,
        vector=vector,
        metadata={
            "source_id": source_id,
            "filename": f"{source_id}.txt",
            "chunk_index": chunk_index,
        },
    )


def test_init_creates_dir_and_empty(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert (tmp_path / "db").is_dir()
    assert store.count() == 0


def test_upsert_empty_batch_returns_zero(tmp_path: Path) -> None:
    assert make_store(tmp_path).upsert_chunks([]) == 0


def test_upsert_and_idempotent_overwrite(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.upsert_chunks([rec("a_0", "文本甲", "a"), rec("a_1", "文本乙", "a", 1)]) == 2
    assert store.count() == 2
    # 相同 ID 覆盖，不产生重复
    store.upsert_chunks([rec("a_0", "文本甲覆盖", "a")])
    assert store.count() == 2


def test_query_sorted_by_similarity(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.upsert_chunks([rec("x_0", "苹果", "x"), rec("y_0", "香蕉", "y", 1)])
    hits = store.query(rec("q", "查询", "q").vector, top_k=2)
    assert len(hits) == 2
    assert hits[0].similarity >= hits[1].similarity  # 降序
    assert hits[0].text in {"苹果", "香蕉"}
    assert hits[0].metadata["filename"].endswith(".txt")
    assert -1.001 <= hits[0].similarity <= 1.001


def test_query_where_filter_with_top_k_exceeding_filtered(tmp_path: Path) -> None:
    """回归：where 过滤后可用条数 < top_k 时不得报错（先算过滤后条数）。"""
    store = make_store(tmp_path)
    store.upsert_chunks(
        [rec("a_0", "甲", "a"), rec("b_0", "乙", "b", 1), rec("b_1", "丙", "b", 2)]
    )
    hits = store.query(
        rec("q", "查询", "q").vector, top_k=5, where={"source_id": "a"}
    )
    assert len(hits) == 1
    assert hits[0].metadata["source_id"] == "a"


def test_query_empty_store_or_zero_top_k(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.query([0.0] * DIM, top_k=3) == []
    store.upsert_chunks([rec("a_0", "甲", "a")])
    assert store.query(rec("q", "x", "q").vector, top_k=0) == []


def test_delete_source(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.upsert_chunks(
        [rec("a_0", "甲", "a"), rec("a_1", "乙", "a", 1), rec("b_0", "丙", "b", 2)]
    )
    assert store.delete_source("nope") == 0  # 来源不存在返回 0
    assert store.delete_source("a") == 2
    assert store.count() == 1


def test_list_sources_grouped_and_sorted(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.upsert_chunks(
        [rec("b_0", "乙", "b"), rec("a_0", "甲一", "a", 3), rec("a_1", "甲二", "a", 4)]
    )
    sources = store.list_sources()
    assert [s.source_id for s in sources] == ["a", "b"]  # 按文件名排序
    by_id = {s.source_id: s for s in sources}
    assert by_id["a"].chunk_count == 2
    assert by_id["b"].chunk_count == 1
    assert by_id["a"].filename == "a.txt"


def test_dim_mismatch_on_reopen(tmp_path: Path) -> None:
    """集合非空且维度与配置不一致时，重开即报错（防脏数据）。"""
    store = make_store(tmp_path)
    store.upsert_chunks([rec("a_0", "甲", "a")])
    with pytest.raises(VectorStoreError, match="维度冲突"):
        make_store(tmp_path, dim=DIM + 3)
