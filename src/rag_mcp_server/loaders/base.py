"""已加载文档的纯数据表示。

所有 loader 实现都应返回此结构，供上层 indexing 流水线消费。
metadata 字段的键名约定：
- source: 文件绝对路径（用于生成 source_id 幂等索引）
- filename: 仅文件名
- extension: 小写扩展名（含点，如 .pdf）
- 其他可选键（pdf 的 page_count 等）可按格式自由扩展
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedDocument:
    """loader 输出：纯文本 + 元数据。"""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
