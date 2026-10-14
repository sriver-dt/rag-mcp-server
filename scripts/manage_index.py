"""运维脚本：手动索引文档 / 查看向量库索引（不依赖 MCP 客户端）。

用法（项目根目录下，使用项目虚拟环境解释器）：
    python scripts/manage_index.py add <文件或目录> [--no-recursive]
    python scripts/manage_index.py list

配置读取与 MCP 服务完全一致（.env / 环境变量），无需额外配置：
- 未配置 EMBEDDING_API_KEY 时 add 会报错，list 照常可用
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_mcp_server.config.settings import get_settings
from rag_mcp_server.tools.context import create_tool_context


def cmd_add(args: argparse.Namespace) -> int:
    """索引单个文件或整个目录。"""
    settings = get_settings()
    ctx = create_tool_context(settings)  # 只创建 ChromaStore，缺 Key 不影响
    try:
        pipeline = ctx.get_pipeline()  # 缺 EMBEDDING_API_KEY 在此报错
    except Exception as exc:
        print(f"[错误] {exc}")
        return 1

    target = Path(args.path)
    if target.is_dir():
        report = pipeline.index_directory(target, recursive=not args.no_recursive)
        print(
            f"目录索引完成: 成功 {report.indexed} / 失败 {report.failed} / "
            f"跳过 {report.skipped}，共写入 {report.total_chunks} 块"
        )
        for item in report.files:
            if item.status == "ok":
                print(f"  [成功] {item.path} ({item.chunk_count} 块)")
            elif item.status == "error":
                print(f"  [失败] {item.path} -> {item.error}")
            else:
                print(f"  [跳过] {item.path}（格式不支持）")
    elif target.is_file():
        result = pipeline.index_document(target)
        print(
            f"索引完成: {result.filename} -> "
            f"source_id={result.source_id}, {result.chunk_count} 块"
        )
    else:
        print(f"[错误] 路径不存在: {target}")
        return 1
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    """查看向量库当前索引情况。"""
    settings = get_settings()
    ctx = create_tool_context(settings)
    sources = ctx.store.list_sources()
    total = ctx.store.count()

    print(f"向量库目录: {settings.chroma_persist_path}")
    print(f"集合名称:   {settings.chroma_collection}")
    print(f"向量总数:   {total}    来源数: {len(sources)}")
    if not sources:
        print("（暂无已索引文档）")
        return 0

    print()
    print(f"{'source_id':<18} {'块数':>4}  {'最近索引时间':<19} 文件名")
    print("-" * 78)
    for s in sources:
        print(f"{s.source_id:<18} {s.chunk_count:>4}  {s.indexed_at:<19} {s.filename}")
        print(f"{'':<18} {'':>4}  {'':<19} {s.source}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 索引运维脚本")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="索引文件或目录")
    p_add.add_argument("path", help="文件或目录路径")
    p_add.add_argument(
        "--no-recursive", action="store_true", help="目录索引时不递归子目录"
    )
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="查看向量库索引")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
