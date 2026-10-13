"""文档管理类 MCP 工具：add_document / list_sources / delete_source。

实现模式：模块级 ``*_impl`` 异步函数承载业务逻辑（返回 dict、异常上抛，
便于直接测试），注册的薄封装负责调用 ``tool_safe`` 统一转 JSON。

同步阻塞说明：文档解析、embedding API、Chroma 读写均为阻塞操作，
统一经 asyncio.to_thread 移出事件循环线程执行。
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from rag_mcp_server.common.exceptions import LoaderError
from rag_mcp_server.tools.context import ToolContext, tool_safe

# ---------- 业务实现（可独立测试） ----------


async def add_document_impl(
    ctx: ToolContext, path: str, recursive: bool = False
) -> dict[str, Any]:
    """索引单个文件或目录。

    pipeline 构建也放入线程池执行：其内部会触发 get_embedder()，
    缺 Key 时抛 EmbeddingError（索引类工具本来就需要 embedder，
    该错误经 tool_safe 返回明确的配置提示）。
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise LoaderError(f"路径不存在: {p}")
    pipeline = await asyncio.to_thread(ctx.get_pipeline)
    if p.is_dir():
        report = await asyncio.to_thread(
            pipeline.index_directory, p, recursive=recursive
        )
        return {
            "ok": True,
            "mode": "directory",
            "indexed": report.indexed,
            "failed": report.failed,
            "skipped": report.skipped,
            "total_chunks": report.total_chunks,
            "files": [asdict(f) for f in report.files],
        }
    result = await asyncio.to_thread(pipeline.index_document, p)
    return {"ok": True, "mode": "file", **asdict(result)}


async def list_sources_impl(ctx: ToolContext) -> dict[str, Any]:
    """列出全部已索引来源。"""
    sources = await asyncio.to_thread(ctx.store.list_sources)
    return {
        "ok": True,
        "total": len(sources),
        "sources": [asdict(s) for s in sources],
    }


async def delete_source_impl(ctx: ToolContext, source_id: str) -> dict[str, Any]:
    """删除指定来源的全部向量。

    直接操作 store 而非走 pipeline：删除不依赖 embedder，
    以保持"无 EMBEDDING_API_KEY 也可用"的设计语义。
    """
    deleted = await asyncio.to_thread(ctx.store.delete_source, source_id)
    return {"ok": True, "source_id": source_id, "deleted": deleted}


# ---------- MCP 工具注册 ----------


def register_document_tools(server: MCPServer, ctx: ToolContext) -> None:
    """向 MCPServer 注册文档管理类工具。"""

    @server.tool()
    async def add_document(path: str, recursive: bool = False) -> str:
        """索引单个文件或整个目录到知识库。

        支持格式：.txt / .md / .pdf / .docx / .html / .htm。
        重复索引同一文件会自动覆盖旧内容（幂等）。

        Args:
            path: 文件或目录的绝对路径（推荐）或相对路径。
            recursive: 当 path 为目录时是否递归索引子目录，默认 False。
        """
        return await tool_safe(add_document_impl(ctx, path, recursive))

    @server.tool()
    async def list_sources() -> str:
        """列出知识库中所有已索引的文档来源。

        返回每个来源的 source_id、文件名、分块数量与最近索引时间，
        source_id 可用于 delete_source 删除操作。
        """
        return await tool_safe(list_sources_impl(ctx))

    @server.tool()
    async def delete_source(source_id: str) -> str:
        """删除知识库中指定来源（文档）的全部向量数据。

        Args:
            source_id: 来源 ID，通过 list_sources 工具获取。
        """
        return await tool_safe(delete_source_impl(ctx, source_id))


__all__ = [
    "add_document_impl",
    "delete_source_impl",
    "list_sources_impl",
    "register_document_tools",
]
