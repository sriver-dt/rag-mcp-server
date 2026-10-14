"""服务可观测性：客户端连接日志中间件。

ConnectionLogMiddleware 注册到 MCPServer.middleware（SDK 官方扩展点，
包装每个入站请求/通知）。拦截 initialize 握手请求，从原始 params 中
提取 clientInfo（客户端名称/版本）与协商协议版本，打印连接日志；
其余请求原样透传。

注意：
- initialize 由 SDK 内联处理，中间件内禁止 await 服务端主动请求
  （会死锁连接）；本中间件只做同步日志输出，天然安全
- stdio 传输下 stdout 是 JSON-RPC 协议通道，日志统一走项目 logger
  （控制台 handler 输出到 stderr，不会污染协议流）
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mcp.server.context import (
    CallNext,
    HandlerResult,
    ServerRequestContext,
)

from rag_mcp_server.common.logging import get_logger


def _extract_client_info(params: Mapping[str, Any] | None) -> tuple[str, str]:
    """从 initialize 的原始 params 提取客户端标识与协议版本。

    params 尚未经 SDK 校验（原始 JSON dict），clientInfo 形如
    ``{"name": "...", "version": "..."}``；字段缺失时返回占位值，
    保证任何畸形握手都能记日志且不影响正常握手流程。
    """
    if not isinstance(params, Mapping):
        return "未知客户端", "未知"
    info = params.get("clientInfo")
    if isinstance(info, Mapping) and info.get("name"):
        client = f"{info['name']} v{info.get('version', '?')}"
    else:
        client = "未知客户端"
    version = str(params.get("protocolVersion") or "未知")
    return client, version


class ConnectionLogMiddleware:
    """记录客户端连接（initialize 握手）的观测中间件。"""

    async def __call__(
        self,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        # request_id 为 None 表示通知，initialize 只会以请求形式出现
        if ctx.method == "initialize" and ctx.request_id is not None:
            client, version = _extract_client_info(ctx.params)
            get_logger().info("客户端已连接: %s（协议版本 %s）", client, version)
        return await call_next(ctx)
