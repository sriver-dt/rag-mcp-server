"""Word 文档加载器（python-docx）。

按文档原始顺序遍历段落与表格（iter_inner_content，需 python-docx >= 1.1.0），
保证表格出现在其上下文位置而非被集中到文末：
- 段落：直接保留原文
- 表格：单元格间以制表符分隔，行与行间换行，跨元素以空行分隔
"""

from __future__ import annotations

from pathlib import Path

from rag_mcp_server.common.exceptions import LoaderError
from rag_mcp_server.loaders.base import LoadedDocument
from rag_mcp_server.loaders.files import _base_metadata


def load_docx(path: Path) -> LoadedDocument:
    """解析 .docx 为纯文本（保留段落/表格原始顺序）。"""
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise LoaderError(
            "python-docx 未安装或版本过低（需 >= 1.1.0 以支持顺序遍历），"
            "执行 pip install -e \".[loaders]\" 后重试"
        ) from exc

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise LoaderError(f"Word 文档打开失败: {path} -> {exc}") from exc

    parts: list[str] = []
    para_count = 0
    table_count = 0

    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if text:
                parts.append(text)
                para_count += 1
        elif isinstance(item, Table):
            table_texts: list[str] = []
            for row in item.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_str = "\t".join(cells)
                if row_str.strip():
                    table_texts.append(row_str)
            if table_texts:
                parts.append("\n".join(table_texts))
                table_count += 1

    metadata = _base_metadata(path)
    metadata["paragraph_count"] = para_count
    metadata["table_count"] = table_count
    return LoadedDocument(text="\n\n".join(parts), metadata=metadata)
