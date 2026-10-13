"""日志初始化：控制台 + 文件双输出。

- 控制台：简洁格式，便于终端观察
- 文件：毫秒级时间戳，写入 ``logs/rag_mcp.log``，超过 10MB 自动轮转
  （保留 5 个备份），避免长期运行日志无限增长
- 重复调用安全：已初始化时直接返回现有 logger，不叠加 handler
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "rag_mcp_server"
_LOG_FILENAME = "rag_mcp.log"
_MAX_LOG_BYTES = 10 * 1024 * 1024  # 单文件上限 10MB
_BACKUP_COUNT = 5  # 轮转备份保留数量


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """初始化项目 logger。

    Args:
        log_dir: 日志目录（通常来自配置 ``LOG_DIR``），不存在时自动创建。
        level: 日志级别字符串（DEBUG/INFO/WARNING/ERROR），非法值回退为 INFO。

    Returns:
        配置完成的 ``rag_mcp_server`` logger。
    """
    logger = logging.getLogger(_LOGGER_NAME)

    # 幂等：避免重复初始化导致日志重复输出
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False  # 防止向 root logger 重复传播

    # %(msecs)03d 需配合 datefmt 显式声明秒级部分，否则毫秒会出现在分隔符后
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（目录不存在时创建，带轮转）
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / _LOG_FILENAME,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # 日志文件不可写时降级为仅控制台输出，不阻断服务启动
        logger.warning("日志文件初始化失败，仅使用控制台输出: %s", log_dir)

    return logger


def get_logger() -> logging.Logger:
    """获取项目 logger（供各模块使用，未初始化时返回未配置的 logger）。"""
    return logging.getLogger(_LOGGER_NAME)
