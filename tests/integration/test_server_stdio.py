"""服务层集成：create_server 工具注册内省 + stdio 子进程端到端冒烟。

stdio 冒烟以真实子进程启动 MCP 服务（无 Key 配置），
验证 JSON-RPC 握手、list_tools 与工具调用的完整协议链路。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rag_mcp_server.config.settings import Settings
from rag_mcp_server.server import create_server

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TOOLS = {
    "add_document",
    "search_knowledge",
    "ask_knowledge",
    "list_sources",
    "delete_source",
}


def test_create_server_registers_five_tools(settings: Settings) -> None:
    server, ctx = create_server(settings)
    assert ctx.settings is settings
    # mcp 2.3.0 的 _tool_manager.list_tools() 为同步方法
    assert {t.name for t in server._tool_manager.list_tools()} == EXPECTED_TOOLS


async def test_stdio_smoke(tmp_path: Path) -> None:
    """子进程启动服务 → 握手 → list_tools → 调用工具（无 Key 语义）。"""
    env = {
        **os.environ,
        "CHROMA_PERSIST_DIR": str(tmp_path / "chroma"),
        "LOG_DIR": str(tmp_path / "logs"),
        "EMBEDDING_API_KEY": "",
        "LLM_API_KEY": "",
        "MCP_TRANSPORT": "stdio",
        "MCP_LOG_LEVEL": "ERROR",
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rag_mcp_server.server"],
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await asyncio.wait_for(session.initialize(), timeout=60)
            assert init.server_info.name == "rag-mcp-server"

            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == EXPECTED_TOOLS

            # list_sources：空库可用
            result = await session.call_tool("list_sources", {})
            payload = json.loads(result.content[0].text)
            assert payload["ok"] is True
            assert payload["sources"] == []

            # add_document：无 Key → 明确配置错误（而非内部崩溃）
            sample = tmp_path / "a.txt"
            sample.write_text("内容", encoding="utf-8")
            result = await session.call_tool("add_document", {"path": str(sample)})
            payload = json.loads(result.content[0].text)
            assert payload["ok"] is False
            assert "EMBEDDING_API_KEY" in payload["error"]
