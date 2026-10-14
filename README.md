# RAG MCP Server

基于 **MCP（Model Context Protocol）** 的独立检索增强生成（RAG）服务。服务以 MCP 工具的形式对外暴露文档索引与知识库问答能力，供支持 MCP 的 AI 客户端（如 IDE Agent、聊天助手等）接入调用。

> 当前状态：**已实现并可运行**（5 个 MCP 工具 + 78 项测试全通过）。个人使用场景，单进程 stdio 部署。

## 技术栈

| 层次 | 选型 | 说明 |
| --- | --- | --- |
| MCP 服务框架 | Python + MCPServer（`mcp` 官方 SDK 2.x） | 注册并对外暴露 RAG 相关工具（2.x 中 FastMCP 已更名为 MCPServer） |
| 向量数据库 | Chroma（本地持久化，cosine） | 零外部服务依赖，数据落盘到 `data/chroma_db/` |
| 文档处理 | pypdf / python-docx / beautifulsoup4 + markdownify | 支持 `.txt` / `.md` / `.pdf` / `.docx` / `.html` |
| Embedding | OpenAI 兼容 API（`/v1/embeddings`） | 经 `EMBEDDING_BASE_URL` 对接任意兼容网关 |
| 答案生成 | OpenAI 兼容 LLM API | 未配置或调用失败时自动降级为返回检索片段 |
| Python 版本 | 3.11+ | — |

## 目录结构

```
rag-mcp-server/
├── pyproject.toml                 # 项目元数据与依赖（src 布局；pytest/ruff/mypy 配置）
├── requirements.txt               # 与 pyproject 保持一致的依赖清单
├── .env.example                   # 环境变量模板（24 个键）
├── issue-log.md                   # 阶段审查记录
├── README.md                      # 项目说明文档（本文件）
│
├── src/
│   └── rag_mcp_server/            # 主源码包
│       ├── server/                # MCP 服务入口：组装依赖、注册工具、按 transport 启动
│       ├── tools/                 # 5 个 MCP 工具：add_document / search_knowledge /
│       │                          #   ask_knowledge / list_sources / delete_source
│       ├── indexing/              # 索引流水线：加载 → 分块 → 嵌入 → 入库（幂等）
│       ├── loaders/               # 文档加载（txt/md/pdf/docx/html，按扩展名分发）
│       ├── chunking/              # 文本分块：段落优先 + 句切 + 硬切 + overlap
│       ├── embeddings/            # Embedding 客户端（协议 + OpenAI 兼容实现）
│       ├── vectorstore/           # Chroma 封装：upsert / query / delete / list_sources
│       ├── retriever/             # 向量检索（Top-K；rerank 预留钩子，默认关闭）
│       ├── prompts/               # RAG 提示词模板（引用格式 [n](文件名)）
│       ├── config/                # 配置加载（pydantic-settings，聚合校验）
│       └── common/                # 公共组件：异常体系、日志（轮转文件 + 控制台）
│
├── data/                          # 运行时数据（不入库，仅保留目录占位）
│   ├── raw/                       # 原始文档：待索引的源文件
│   ├── processed/                 # 中间产物
│   └── chroma_db/                 # Chroma 本地持久化目录
│
├── tests/
│   ├── conftest.py                # 共享夹具：FakeEmbedder（哈希向量）+ 临时配置
│   ├── unit/                      # 单元测试：settings/logging/chunking/loaders/
│   │                              #   chroma_store/pipeline/retriever/prompts/tools
│   └── integration/               # 集成测试：全链路 + stdio 子进程冒烟
│
├── scripts/                       # 运维脚本：manage_index.py（命令行索引 / 查看索引）
├── docs/                          # 项目文档（预留）
└── logs/                          # 运行日志（不入库，10MB 轮转 × 5 备份）
```

## 数据流

```
文档路径 (add_document)
    │  loaders/      按扩展名解析为纯文本 + 元数据
    ▼
文本 + 元数据
    │  chunking/     段落优先分块（超长按句切，相邻块 overlap）
    ▼
文本块 (chunks)
    │  embeddings/   OpenAI 兼容 API 批量向量化
    ▼
向量 + 文本 + 元数据
    │  vectorstore/  幂等写入 Chroma（source_id = 路径 SHA256 前 16 位）
    ▼
用户提问 ──► retriever/ 向量检索 Top-K ──► prompts/ 组装上下文 ──► LLM 生成带引用答案
              ▲                                │（LLM 未配置/失败时降级）
              └── tools/ 以 MCP 工具形式对外提供 ─┘
```

## 快速开始

```powershell
# 1. 创建虚拟环境（需系统独立安装的 Python 3.11+）
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell

# 2. 安装依赖（含文档解析可选依赖与开发工具）
pip install -e ".[dev,loaders]"

# 3. 配置环境变量：复制 .env.example 为 .env，至少配置：
#    EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL / EMBEDDING_DIM
#    LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（不配置则 ask_knowledge 降级）

# 4. 启动 MCP 服务（stdio 传输，默认）
python -m rag_mcp_server.server
# 或安装后的命令行入口
rag-mcp-server
```

MCP 客户端接入示例（如 IDE 的 MCP 配置）：

```json
{
  "mcpServers": {
    "rag-mcp-server": {
      "command": "e:\\trae\\workspace\\rag-mcp-server\\.venv\\Scripts\\python.exe",
      "args": ["-m", "rag_mcp_server.server"]
    }
  }
}
```

## MCP 工具（5 个）

| 工具 | 功能 | 依赖 |
| --- | --- | --- |
| `add_document(path, recursive?)` | 索引文件或目录（幂等：重复索引自动覆盖） | indexing |
| `search_knowledge(query, top_k?)` | 纯检索，返回最相关原文片段（不调 LLM） | retriever |
| `ask_knowledge(question, top_k?)` | RAG 问答：检索 → prompt → LLM 生成带 `[n](文件名)` 引用的回答 | retriever + LLM |
| `list_sources()` | 列出已索引来源（source_id/文件名/分块数/时间） | vectorstore |
| `delete_source(source_id)` | 删除指定来源的全部向量 | vectorstore |

## 命令行运维脚本

不依赖 MCP 客户端，直接在命令行索引文档或查看向量库状态。脚本与 MCP 服务读取同一份 `.env` 配置，无需额外配置（需在项目虚拟环境激活后运行）：

```powershell
# 索引单个文件或目录（目录默认递归子目录，自动按扩展名过滤）
python scripts/manage_index.py add <文件或目录路径>

# 目录索引时不递归子目录
python scripts/manage_index.py add data\raw --no-recursive

# 查看向量库索引（来源列表、各来源分块数、最近索引时间、完整路径）
python scripts/manage_index.py list
```

说明：

- `add` 需要 `EMBEDDING_API_KEY`，未配置时报明确的配置错误；`list` 不依赖任何 Key，随时可用
- `add` 幂等：重复索引同一路径会自动覆盖旧索引，无需先删除
- 目录批量索引逐文件容错，单文件失败不中断，结尾输出成功/失败/跳过汇总

## 降级语义（无 Key 也可用）

- 未配置 `EMBEDDING_API_KEY`：服务正常启动，`list_sources` / `delete_source` 可用；索引与检索类工具返回明确的配置错误提示
- 未配置 `LLM_API_KEY` / openai 未安装 / LLM 调用失败：`ask_knowledge` 降级返回检索片段原文（`degraded: true`），不中断会话

## 测试与质量门禁

```powershell
pytest                      # 78 项测试（单元 + 集成，FakeEmbedder 不出网）
ruff check .                # 代码检查
mypy src                    # 严格类型检查
python -m compileall -q src tests
```

集成测试包含 stdio 端到端冒烟：以子进程真实启动服务，完成 MCP 握手、`list_tools` 与工具调用验证。

## 版本控制

- 默认分支：`main`
- `data/` 下的上传文档、中间产物、Chroma 索引以及 `logs/`、`.venv/`、缓存目录均已在 `.gitignore` 中忽略
- `.env` 等密钥文件已忽略；配置请以 `.env.example` 为模板
- 文本文件统一使用 LF 换行；Windows 脚本（`.ps1/.bat/.cmd`）保持 CRLF
