"""工具层共享上下文与结果格式化。

ToolContext 采用惰性构建：服务启动只创建 ChromaStore（无需 API Key），
embedder 首次被索引/检索类工具使用时才创建——未配置 EMBEDDING_API_KEY 时
服务仍可启动，list_sources / delete_source 照常可用。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rag_mcp_server.common.exceptions import RagMcpError
from rag_mcp_server.config.settings import Settings
from rag_mcp_server.embeddings.base import EmbeddingClient
from rag_mcp_server.indexing.pipeline import IndexingPipeline
from rag_mcp_server.retriever.retriever import Retriever
from rag_mcp_server.vectorstore.chroma_store import ChromaStore

if TYPE_CHECKING:
    from openai import OpenAI


def tool_result_json(data: dict[str, Any]) -> str:
    """将工具结果序列化为 JSON 字符串（中文原样输出）。"""
    return json.dumps(data, ensure_ascii=False, indent=2)


async def tool_safe(awaitable: Awaitable[dict[str, Any]]) -> str:
    """执行业务协程并统一转换异常为 {"ok": false, "error": ...} 结果。

    RagMcpError 输出原始消息；未知异常输出简化的内部错误提示，
    避免向 MCP 客户端暴露堆栈细节。
    """
    try:
        return tool_result_json(await awaitable)
    except RagMcpError as exc:
        return tool_result_json({"ok": False, "error": str(exc)})
    except Exception as exc:  # 兜底：不让异常逃逸中断 MCP 会话
        return tool_result_json({"ok": False, "error": f"内部错误: {exc}"})


@dataclass
class ToolContext:
    """MCP 工具共享上下文：配置 + 惰性业务组件。"""

    settings: Settings
    store: ChromaStore
    # 测试可注入假 embedder；生产为 None 时按需创建 OpenAI 兼容客户端
    embedder: EmbeddingClient | None = None
    _pipeline: IndexingPipeline | None = field(
        default=None, repr=False, init=False
    )
    _retriever: Retriever | None = field(default=None, repr=False, init=False)
    _llm_client: OpenAI | None = field(default=None, repr=False, init=False)

    def get_embedder(self) -> EmbeddingClient:
        """获取向量客户端（未注入时惰性创建，缺 Key 抛 EmbeddingError）。"""
        if self.embedder is None:
            from rag_mcp_server.embeddings import OpenAICompatibleEmbedder

            self.embedder = OpenAICompatibleEmbedder(self.settings)
        return self.embedder

    def get_pipeline(self) -> IndexingPipeline:
        """获取索引流水线（惰性单例）。"""
        if self._pipeline is None:
            self._pipeline = IndexingPipeline(
                self.settings, self.get_embedder(), self.store
            )
        return self._pipeline

    def get_retriever(self) -> Retriever:
        """获取检索器（惰性单例）。"""
        if self._retriever is None:
            self._retriever = Retriever(
                self.settings, self.get_embedder(), self.store
            )
        return self._retriever

    def get_llm_client(self) -> OpenAI:
        """获取 LLM 客户端（惰性单例，连接池复用）。

        openai 延迟导入：未安装 openai 不影响模块加载，
        只有真正调用 LLM 的 ask_knowledge 会收到明确错误。
        """
        if self._llm_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RagMcpError("openai 库未安装，无法调用 LLM") from exc
            self._llm_client = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._llm_client


def create_tool_context(settings: Settings) -> ToolContext:
    """按配置构建工具上下文（ChromaStore 启动即创建，含维度校验）。"""
    store = ChromaStore(
        persist_dir=settings.chroma_persist_path,
        collection_name=settings.chroma_collection,
        expected_dim=settings.embedding_dim,
    )
    return ToolContext(settings=settings, store=store)
