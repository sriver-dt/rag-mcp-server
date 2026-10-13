"""项目统一异常体系。

所有自定义异常继承自 RagMcpError，便于在 MCP 工具层统一捕获并转换为
用户可读的错误消息，而不暴露内部堆栈细节。
"""


class RagMcpError(Exception):
    """RAG MCP 服务的基础异常，所有自定义异常的父类。"""


class ConfigError(RagMcpError):
    """配置缺失或非法（如缺少 API Key、参数取值超出约束）。"""


class LoaderError(RagMcpError):
    """文档加载/解析失败（格式不支持、文件损坏、编码错误等）。"""


class EmbeddingError(RagMcpError):
    """Embedding 调用失败（网络错误、API 拒绝、维度不匹配等）。"""


class VectorStoreError(RagMcpError):
    """向量库操作失败（Chroma 读写、集合操作等）。"""


class RetrievalError(RagMcpError):
    """检索阶段失败（查询向量化、相似度检索等）。"""
