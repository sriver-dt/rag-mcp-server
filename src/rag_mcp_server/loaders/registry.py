"""加载器注册表：按扩展名分发到具体实现。

新增格式时只需：
1. 在 loaders/ 下新增实现函数（接收 Path、返回 LoadedDocument）
2. 在 _HANDLERS 映射里添加扩展名到函数的绑定
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_mcp_server.common.exceptions import LoaderError
from rag_mcp_server.loaders.base import LoadedDocument
from rag_mcp_server.loaders.docx import load_docx
from rag_mcp_server.loaders.files import load_plain_text
from rag_mcp_server.loaders.html import load_html
from rag_mcp_server.loaders.pdf import load_pdf

# 扩展名（含点、小写）→ 加载函数
_HANDLERS: dict[str, Callable[[Path], LoadedDocument]] = {
    ".txt": load_plain_text,
    ".md": load_plain_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(_HANDLERS.keys()))


def load_document(path: Path) -> LoadedDocument:
    """按扩展名自动选择加载器。

    Args:
        path: 待解析文件的绝对或相对路径。

    Raises:
        LoaderError: 文件不存在、扩展名不支持或解析失败。
    """
    if not path.is_file():
        raise LoaderError(f"文件不存在: {path}")

    ext = path.suffix.lower()
    handler = _HANDLERS.get(ext)
    if handler is None:
        raise LoaderError(
            f"不支持的文件类型 {ext!r}，支持的扩展名: {SUPPORTED_EXTENSIONS}"
        )
    return handler(path)
