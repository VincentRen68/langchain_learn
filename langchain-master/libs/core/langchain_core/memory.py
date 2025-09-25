"""**记忆**维护链的状态，整合来自过去运行的上下文。

此模块包含来自 LangChain v0.0.x 的记忆抽象。

这些抽象现在已被弃用，将在 LangChain v1.0.0 中移除。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import ConfigDict

from langchain_core._api import deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.runnables import run_in_executor


@deprecated(
    since="0.3.3",
    removal="1.0.0",
    message=(
        "请查看迁移指南："
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class BaseMemory(Serializable, ABC):
    """链中记忆的抽象基类。

    记忆指的是链中的状态。记忆可用于存储关于链过去执行的信息，
        并将该信息注入到链的未来执行的输入中。例如，对于对话链，
        记忆可用于存储对话并自动将它们添加到未来的模型提示中，
        以便模型具有必要的上下文来连贯地响应最新的输入。

    示例:
        .. code-block:: python

            class SimpleMemory(BaseMemory):
                memories: dict[str, Any] = dict()

                @property
                def memory_variables(self) -> list[str]:
                    return list(self.memories.keys())

                def load_memory_variables(
                    self, inputs: dict[str, Any]
                ) -> dict[str, str]:
                    return self.memories

                def save_context(
                    self, inputs: dict[str, Any], outputs: dict[str, str]
                ) -> None:
                    pass

                def clear(self) -> None:
                    pass

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @abstractmethod
    def memory_variables(self) -> list[str]:
        """此记忆类将添加到链输入中的字符串键。"""

    @abstractmethod
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据链的文本输入返回键值对。

        参数:
            inputs: 链的输入。

        返回:
            键值对的字典。
        """

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据链的文本输入异步返回键值对。

        参数:
            inputs: 链的输入。

        返回:
            键值对的字典。
        """
        return await run_in_executor(None, self.load_memory_variables, inputs)

    @abstractmethod
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将此次链运行的上下文保存到记忆中。

        参数:
            inputs: 链的输入。
            outputs: 链的输出。
        """

    async def asave_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """将此次链运行的上下文异步保存到记忆中。

        参数:
            inputs: 链的输入。
            outputs: 链的输出。
        """
        await run_in_executor(None, self.save_context, inputs, outputs)

    @abstractmethod
    def clear(self) -> None:
        """清除记忆内容。"""

    async def aclear(self) -> None:
        """异步清除记忆内容。"""
        await run_in_executor(None, self.clear)
