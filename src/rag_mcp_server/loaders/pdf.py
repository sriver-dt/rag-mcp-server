"""PDF 加载器（pypdf）。

逐页读取并拼接，每页间以换行分隔；提取标题优先（/Title 元数据），
不可用时回退文件名。空页面自动跳过。
"""

from __future__ import annotations

from pathlib import Path

from rag_mcp_server.common.exceptions import LoaderError
from rag_mcp_server.loaders.base import LoadedDocument
from rag_mcp_server.loaders.files import _base_metadata


def load_pdf(path: Path) -> LoadedDocument:
    """解析 PDF 为纯文本。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise LoaderError(
            "pypdf 未安装，执行 pip install -e \".[loaders]\" 后重试"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise LoaderError(f"PDF 打开失败: {path} -> {exc}") from exc

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""  # 个别页面损坏时跳过，不阻断整个文档
        if text.strip():
            page_texts.append(text.strip())

    full_text = "\n\n".join(page_texts)

    # 标题优先从 PDF 元数据提取
    title = path.stem
    meta = getattr(reader, "metadata", None)
    if meta:
        try:
            pdf_title = getattr(meta, "title", None) or meta.get("/Title")
            if pdf_title and str(pdf_title).strip():
                title = str(pdf_title).strip()
        except Exception:
            pass

    metadata = _base_metadata(path)
    metadata.update({
        "page_count": len(reader.pages),
        "title": title,
    })
    return LoadedDocument(text=full_text, metadata=metadata)
