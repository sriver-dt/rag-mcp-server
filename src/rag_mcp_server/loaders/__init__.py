"""文档加载器：按扩展名解析不同格式文件为纯文本。"""

from rag_mcp_server.loaders.base import LoadedDocument
from rag_mcp_server.loaders.registry import SUPPORTED_EXTENSIONS, load_document

__all__ = ["LoadedDocument", "load_document", "SUPPORTED_EXTENSIONS"]
