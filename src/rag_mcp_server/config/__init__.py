"""配置模块：全局 Settings 单例。"""

from rag_mcp_server.config.settings import (
    PROJECT_ROOT,
    Settings,
    get_settings,
    reset_settings,
)

__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "reset_settings"]
