"""聊天消息。"""

from typing import Any, Literal

from typing_extensions import override

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class ChatMessage(BaseMessage):
    """可以被赋予任意发言者（即角色）的消息。"""

    role: str
    """消息的发言者/角色。"""

    type: Literal["chat"] = "chat"
    """消息的类型（用于序列化）。默认为 "chat"。"""


class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """聊天消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ChatMessageChunk"] = "ChatMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于序列化）。
    默认为 "ChatMessageChunk"。"""

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, ChatMessageChunk):
            if self.role != other.role:
                msg = "Cannot concatenate ChatMessageChunks with different roles."
                raise ValueError(msg)

            return self.__class__(
                role=self.role,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
            )
        if isinstance(other, BaseMessageChunk):
            return self.__class__(
                role=self.role,
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
