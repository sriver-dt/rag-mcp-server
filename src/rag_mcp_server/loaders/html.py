"""HTML 加载器（beautifulsoup4 + markdownify）。

先剥离 script/style/nav/footer 等非正文标签，
再经 markdownify 转为 Markdown 文本（保留标题/列表/表格结构，
比纯 get_text() 更适合 RAG 分块）。

注意：不移除 <header>——文章页的标题（h1）常位于 header 内，
删除会导致正文丢失主标题；残留的站点导航噪音影响远小于标题丢失。
"""

from __future__ import annotations

from pathlib import Path

from rag_mcp_server.common.exceptions import LoaderError
from rag_mcp_server.loaders.base import LoadedDocument
from rag_mcp_server.loaders.files import _base_metadata

# 转换为 Markdown 前需要移除的标签（脚本/导航/页脚等非正文）
# header 有意保留：见模块 docstring 说明
_REMOVE_TAGS = ("script", "style", "nav", "footer", "aside", "noscript")


def load_html(path: Path) -> LoadedDocument:
    """解析 .html / .htm 为 Markdown 文本。"""
    try:
        from bs4 import BeautifulSoup
        from markdownify import markdownify as md
    except ImportError as exc:
        raise LoaderError(
            "beautifulsoup4 / markdownify 未安装，执行 pip install -e \".[loaders]\" 后重试"
        ) from exc

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in _REMOVE_TAGS:
        for node in soup.find_all(tag):
            node.decompose()

    markdown = md(
        str(soup),
        heading_style="ATX",  # # 风格标题，便于后续按标题切分
    )

    title = path.stem
    title_tag = soup.find("title")
    if title_tag and title_tag.string and title_tag.string.strip():
        title = title_tag.string.strip()

    metadata = _base_metadata(path)
    metadata["title"] = title
    return LoadedDocument(text=markdown.strip(), metadata=metadata)
