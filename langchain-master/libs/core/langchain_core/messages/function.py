"""函数消息。"""

from typing import Any, Literal

from typing_extensions import override

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class FunctionMessage(BaseMessage):
    """用于将执行工具后的结果传递回模型的消息。

    FunctionMessage 是 ToolMessage 模式的一个旧版本，
    它不包含 tool_call_id 字段。

    tool_call_id 字段用于将工具调用请求与工具调用响应关联起来。
    这在聊天模型能够并行请求多个工具调用的情况下非常有用。
    """

    name: str
    """被执行的函数的名称。"""

    type: Literal["function"] = "function"
    """消息的类型（用于序列化）。默认为 "function"。"""


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """函数消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["FunctionMessageChunk"] = "FunctionMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于序列化）。
    默认为 "FunctionMessageChunk"。"""

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                msg = "Cannot concatenate FunctionMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                name=self.name,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
            )

        return super().__add__(other)
