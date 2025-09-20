from typing import Any

from langchain_core.memory import BaseMemory


class ReadOnlySharedMemory(BaseMemory):
    """一个只读且不可更改的记忆包装器。"""

    memory: BaseMemory

    @property
    def memory_variables(self) -> list[str]:
        """返回内存变量。"""
        return self.memory.memory_variables

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """从内存中加载内存变量。"""
        return self.memory.load_memory_variables(inputs)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """任何内容都不应被保存或更改。"""

    def clear(self) -> None:
        """无需清除任何内容，记忆像保险库一样安全。"""
