from typing import Any

from langchain_core.memory import BaseMemory
from typing_extensions import override


class SimpleMemory(BaseMemory):
    """简单的记忆。

    用于存储上下文或其他在提示之间不应改变的信息的简单记忆。
    """

    memories: dict[str, Any] = {}

    @property
    @override
    def memory_variables(self) -> list[str]:
        return list(self.memories.keys())

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        return self.memories

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """任何内容都不应被保存或更改，我的记忆是固定不变的。"""

    def clear(self) -> None:
        """无需清除任何内容，记忆像保险库一样安全。"""
