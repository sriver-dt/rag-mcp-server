"""纯文本类加载器：.txt / .md。

UTF-8 优先，失败时回退 gb18030（覆盖简体中文常见编码），
仍失败则按 latin-1 兜底（保证文件能读出来，即使乱码）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_mcp_server.common.exceptions import LoaderError
from rag_mcp_server.loaders.base import LoadedDocument

_ENCODING_FALLBACKS = ("utf-8", "gb18030", "latin-1")


def load_plain_text(path: Path) -> LoadedDocument:
    """按多编码回退顺序读取 .txt / .md 文件。"""
    last_error: UnicodeDecodeError | None = None
    for enc in _ENCODING_FALLBACKS:
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise LoaderError(
            f"无法解码文件（已尝试 {_ENCODING_FALLBACKS}）: {path}"
        ) from last_error
    return LoadedDocument(
        text=text,
        metadata=_base_metadata(path),
    )


def _base_metadata(path: Path) -> dict[str, Any]:
    """各 loader 通用的基础元数据（source / filename / extension）。"""
    return {
        "source": str(path.resolve()),
        "filename": path.name,
        "extension": path.suffix.lower(),
    }
