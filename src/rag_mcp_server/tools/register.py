"""MCP 工具注册入口：聚合全部工具到 MCPServer。"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from rag_mcp_server.tools.context import ToolContext
from rag_mcp_server.tools.document_tools import register_document_tools
from rag_mcp_server.tools.search_tools import register_search_tools


def register_tools(server: MCPServer, ctx: ToolContext) -> None:
    """向 MCPServer 注册全部 5 个 RAG 工具。

    add_document / list_sources / delete_source 属文档管理，
    search_knowledge / ask_knowledge 属检索问答。
    """
    register_document_tools(server, ctx)
    register_search_tools(server, ctx)
