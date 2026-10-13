"""日志初始化测试：幂等、落盘、非法级别回退。"""

from __future__ import annotations

import logging
from pathlib import Path

from rag_mcp_server.common.logging import get_logger, setup_logging


def test_setup_logging_creates_file_and_idempotent(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logger = setup_logging(log_dir, "INFO")
    count_first = len(logger.handlers)
    assert count_first >= 2  # 控制台 + 文件

    # 重复初始化：同一 logger、不叠加 handler
    assert setup_logging(log_dir, "DEBUG") is logger
    assert len(logger.handlers) == count_first

    logger.info("日志写入测试消息")
    for handler in logger.handlers:
        handler.flush()
    log_file = log_dir / "rag_mcp.log"
    assert log_file.exists()
    assert "日志写入测试消息" in log_file.read_text(encoding="utf-8")

    # 清理全局 handler（logger 为模块级单例，避免污染其他测试）
    logger.handlers.clear()


def test_invalid_level_falls_back_to_info(tmp_path: Path) -> None:
    logger = setup_logging(tmp_path / "logs2", "NOT_A_LEVEL")
    assert logger.level == logging.INFO
    logger.handlers.clear()


def test_get_logger_name() -> None:
    assert get_logger().name == "rag_mcp_server"
