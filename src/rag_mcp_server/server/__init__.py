"""MCP 服务入口：组装依赖、注册工具、启动传输。

启动流程（main）：
1. 加载配置（.env 校验失败直接退出并给出明确错误）
2. 初始化日志（控制台 + logs/rag_mcp.log 轮转文件）
3. create_server()：构建 ToolContext（ChromaStore 即时创建、embedder 惰性）
   并注册 5 个工具
4. server.run()：按 MCP_TRANSPORT 启动（stdio 为默认；sse/streamable-http
   时附带 host/port）

embedder 惰性设计意味着：未配置 EMBEDDING_API_KEY 时服务也能启动，
list_sources / delete_source 可用，索引与检索类工具会返回明确的
"未配置 Key" 错误提示。
"""

from __future__ import annotations

import sys

from mcp.server.mcpserver import MCPServer

from rag_mcp_server.common.exceptions import ConfigError, RagMcpError
from rag_mcp_server.common.logging import setup_logging
from rag_mcp_server.config.settings import Settings, get_settings
from rag_mcp_server.tools import create_tool_context, register_tools
from rag_mcp_server.tools.context import ToolContext

SERVER_NAME = "rag-mcp-server"
SERVER_VERSION = "0.1.0"
SERVER_INSTRUCTIONS = (
    "本地知识库 RAG 服务。可用工具：\n"
    "- add_document: 索引文件/目录（支持 txt/md/pdf/docx/html）\n"
    "- search_knowledge: 检索相关原文片段\n"
    "- ask_knowledge: 基于知识库的 RAG 问答（带引用）\n"
    "- list_sources: 查看已索引文档\n"
    "- delete_source: 删除指定文档的全部向量"
)


def create_server(settings: Settings | None = None) -> tuple[MCPServer, ToolContext]:
    """构建 MCPServer 实例并注册全部工具。

    Returns:
        (server, ctx) 元组：ctx 暴露给测试与高级用法（如直接调用 impl）。
    """
    settings = settings or get_settings()
    ctx = create_tool_context(settings)

    server = MCPServer(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        log_level=settings.mcp_log_level.upper(),  # type: ignore[arg-type]
    )
    register_tools(server, ctx)
    return server, ctx


def main() -> None:
    """服务主入口（pyproject scripts 与 python -m 共用）。

    错误处理顺序说明：配置加载失败时文件日志尚未初始化（log 目录
    本身依赖配置），因此启动阶段的失败输出到 stderr 并以退出码 1
    结束；日志初始化之后的运行期错误交由日志与 MCP 框架处理。
    """
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    setup_logging(settings.log_path, settings.mcp_log_level)

    try:
        server, _ = create_server(settings)
    except RagMcpError as exc:
        # 典型场景：Chroma 集合维度与 EMBEDDING_DIM 不一致
        print(f"服务初始化失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # 按传输方式显式分支（匹配 run() 重载签名；stdio 无监听参数）。
    # 配置校验已将 MCP_TRANSPORT 限制为三个取值，else 分支是面向未来
    # 新增 transport 时的显式防御，避免误当 streamable-http 启动。
    transport = settings.mcp_transport
    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "sse":
        server.run(
            transport="sse",
            host=settings.mcp_host,
            port=settings.mcp_port,
        )
    elif transport == "streamable-http":
        server.run(
            transport="streamable-http",
            host=settings.mcp_host,
            port=settings.mcp_port,
        )
    else:
        print(f"不支持的 MCP_TRANSPORT: {transport!r}", file=sys.stderr)
        raise SystemExit(1)
