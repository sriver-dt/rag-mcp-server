"""全局配置：基于 pydantic-settings，从项目根目录的 .env 读取。

字段与 .env.example 中的 24 个环境变量一一对应：
- MCP_*：服务与传输（4 个）
- LLM_*：答案生成模型（5 个）
- EMBEDDING_*：向量化模型（5 个）
- CHROMA_*：向量库（2 个）
- CHUNK_* / DATA_*：文档处理（4 个）
- RETRIEVAL_*：检索（3 个）
- LOG_DIR：日志目录（1 个）

路径类配置支持相对路径（相对项目根目录），通过只读属性解析为绝对 Path。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_mcp_server.common.exceptions import ConfigError

# 项目根目录：src/rag_mcp_server/config/settings.py 向上 4 级
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# RETRIEVAL_TOP_K 上限：防止误配拖垮检索与 LLM 上下文
_MAX_TOP_K = 50


class Settings(BaseSettings):
    """RAG MCP 服务全局配置（单例，经 get_settings() 获取）。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- MCP ----------
    mcp_log_level: str = "INFO"
    mcp_transport: str = "stdio"  # stdio / sse / streamable-http
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000

    # ---------- LLM ----------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # ---------- Embedding ----------
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    # 单次 embedding 请求的最大输入条数，需不超过服务端限制
    # （如 DashScope 系列模型限 20，OpenAI text-embedding 系列限 2048）
    embedding_batch_size: int = 20

    # ---------- Chroma ----------
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection: str = "rag_docs"

    # ---------- 文档处理 / 分块 ----------
    chunk_size: int = 800
    chunk_overlap: int = 100
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"

    # ---------- 检索 ----------
    retrieval_top_k: int = 5
    rerank_enabled: bool = False
    rerank_model: str = ""

    # ---------- 日志 ----------
    log_dir: str = "logs"

    # ---------- 约束校验（聚合所有错误，一次性抛出） ----------
    @model_validator(mode="after")
    def _validate_constraints(self) -> Settings:
        errors: list[str] = []
        if self.chunk_size <= 0:
            errors.append(f"CHUNK_SIZE 必须为正整数，当前为 {self.chunk_size}")
        if self.chunk_overlap < 0:
            errors.append(f"CHUNK_OVERLAP 不能为负数，当前为 {self.chunk_overlap}")
        if self.chunk_size > 0 and self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"CHUNK_OVERLAP({self.chunk_overlap}) 必须小于 CHUNK_SIZE({self.chunk_size})"
            )
        if self.embedding_dim <= 0:
            errors.append(f"EMBEDDING_DIM 必须为正整数，当前为 {self.embedding_dim}")
        if self.embedding_batch_size <= 0:
            errors.append(
                f"EMBEDDING_BATCH_SIZE 必须为正整数，当前为 {self.embedding_batch_size}"
            )
        if self.retrieval_top_k <= 0:
            errors.append(
                f"RETRIEVAL_TOP_K 必须为正整数，当前为 {self.retrieval_top_k}")
        elif self.retrieval_top_k > _MAX_TOP_K:
            errors.append(
                f"RETRIEVAL_TOP_K 超过上限 {_MAX_TOP_K}，当前为 {self.retrieval_top_k}")
        if self.mcp_transport not in ("stdio", "sse", "streamable-http"):
            errors.append(
                f"MCP_TRANSPORT 非法: {self.mcp_transport!r}，"
                "可选值: stdio / sse / streamable-http"
            )
        if errors:
            raise ConfigError("配置校验失败:\n- " + "\n- ".join(errors))
        return self

    # ---------- 路径解析（相对路径 → 项目根下的绝对路径） ----------
    @property
    def chroma_persist_path(self) -> Path:
        """Chroma 持久化目录（绝对路径）。"""
        return (PROJECT_ROOT / self.chroma_persist_dir).resolve()

    @property
    def data_raw_path(self) -> Path:
        """原始文档目录（绝对路径）。"""
        return (PROJECT_ROOT / self.data_raw_dir).resolve()

    @property
    def data_processed_path(self) -> Path:
        """分块中间产物目录（绝对路径）。"""
        return (PROJECT_ROOT / self.data_processed_dir).resolve()

    @property
    def log_path(self) -> Path:
        """日志目录（绝对路径）。"""
        return (PROJECT_ROOT / self.log_dir).resolve()

    @property
    def llm_configured(self) -> bool:
        """LLM 是否已配置 API Key（决定 ask_knowledge 是否可生成答案）。"""
        return bool(self.llm_api_key.strip())

    @property
    def embedding_configured(self) -> bool:
        """Embedding 是否已配置 API Key（决定索引/检索是否可用）。"""
        return bool(self.embedding_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例（首次调用时读取 .env 并完成校验）。"""
    return Settings()


def reset_settings() -> None:
    """清除配置缓存（仅供测试使用）。"""
    get_settings.cache_clear()
