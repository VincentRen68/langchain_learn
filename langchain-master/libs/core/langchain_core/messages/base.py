"""消息基类。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union, cast

from pydantic import ConfigDict, Field

from langchain_core.load.serializable import Serializable
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessage(Serializable):
    """消息的抽象基类。

    消息是聊天模型（ChatModels）的输入和输出。
    """

    content: Union[str, list[Union[str, dict]]]
    """消息的字符串内容。"""

    additional_kwargs: dict = Field(default_factory=dict)
    """为与消息关联的附加负载数据保留的字段。

    例如，对于来自AI的消息，这里可以包含由模型提供商编码的工具调用（tool calls）。
    """

    response_metadata: dict = Field(default_factory=dict)
    """响应的元数据。例如：响应头、对数概率（logprobs）、令牌计数、模型名称等。"""

    type: str
    """消息的类型。必须是该消息类型唯一的字符串。

    该字段旨在方便反序列化消息时轻松识别其类型。
    """

    name: Optional[str] = None
    """消息的可选名称。

    可用于为消息提供一个人类可读的名称。

    该字段的使用是可选的，是否使用取决于模型的具体实现。
    """

    id: Optional[str] = Field(default=None, coerce_numbers_to_str=True)
    """消息的可选唯一标识符。理想情况下，应由创建此消息的提供商/模型提供。"""

    model_config = ConfigDict(
        extra="allow",
    )

    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        """将 content 作为位置参数传入。

        参数:
            content: 消息的字符串内容。
        """
        super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """BaseMessage 是可序列化的。

        返回:
            True
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 langchain 对象的命名空间。

        返回:
            ``["langchain", "schema", "messages"]``
        """
        return ["langchain", "schema", "messages"]

    def text(self) -> str:
        """获取消息的文本内容。

        返回:
            消息的文本内容。
        """
        if isinstance(self.content, str):
            return self.content

        # must be a list
        blocks = [
            block
            for block in self.content
            if isinstance(block, str)
            or (block.get("type") == "text" and isinstance(block.get("text"), str))
        ]
        return "".join(
            block if isinstance(block, str) else block["text"] for block in blocks
        )

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """将此消息与另一个消息拼接。

        参数:
            other: 要与此消息拼接的另一个消息。

        返回:
            一个包含两条消息的 ChatPromptTemplate。
        """
        # Import locally to prevent circular imports.
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

        prompt = ChatPromptTemplate(messages=[self])
        return prompt + other

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """获取消息的美化表示。

        参数:
            html: 是否将消息格式化为 HTML。如果为 True，消息将使用 HTML 标签格式化。
                默认为 False。

        返回:
            消息的美化表示。
        """
        title = get_msg_title_repr(self.type.title() + " Message", bold=html)
        # TODO: handle non-string content.
        if self.name is not None:
            title += f"\nName: {self.name}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        """打印消息的美化表示。"""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


def merge_content(
    first_content: Union[str, list[Union[str, dict]]],
    *contents: Union[str, list[Union[str, dict]]],
) -> Union[str, list[Union[str, dict]]]:
    """合并多个消息内容。

    参数:
        first_content: 第一个内容。可以是字符串或列表。
        contents: 其他内容。可以是字符串或列表。

    返回:
        合并后的内容。
    """
    merged = first_content
    for content in contents:
        # If current is a string
        if isinstance(merged, str):
            # If the next chunk is also a string, then merge them naively
            if isinstance(content, str):
                merged += content
            # If the next chunk is a list, add the current to the start of the list
            else:
                merged = [merged, *content]
        elif isinstance(content, list):
            # If both are lists
            merged = merge_lists(cast("list", merged), content)  # type: ignore[assignment]
        # If the first content is a list, and the second content is a string
        # If the last element of the first content is a string
        # Add the second content to the last element
        elif merged and isinstance(merged[-1], str):
            merged[-1] += content
        # If second content is an empty string, treat as a no-op
        elif content:
            # Otherwise, add the second content as a new element of the list
            merged.append(content)
    return merged


class BaseMessageChunk(BaseMessage):
    """消息块，可以与其他消息块拼接。"""

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        """消息块支持与其他消息块进行拼接。

        此功能对于将流式模型生成的消息块组合成一个完整的消息非常有用。

        参数:
            other: 要与此消息块拼接的另一个消息块。

        返回:
            一个新的消息块，它是此消息块与另一个消息块的拼接结果。

        异常:
            TypeError: 如果另一个对象不是消息块。

        例如，

        `AIMessageChunk(content="Hello") + AIMessageChunk(content=" World")`

        将得到 `AIMessageChunk(content="Hello World")`
        """
        if isinstance(other, BaseMessageChunk):
            # If both are (subclasses of) BaseMessageChunk,
            # concat into a single BaseMessageChunk

            return self.__class__(
                id=self.id,
                type=self.type,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
            )
        if isinstance(other, list) and all(
            isinstance(o, BaseMessageChunk) for o in other
        ):
            content = merge_content(self.content, *(o.content for o in other))
            additional_kwargs = merge_dicts(
                self.additional_kwargs, *(o.additional_kwargs for o in other)
            )
            response_metadata = merge_dicts(
                self.response_metadata, *(o.response_metadata for o in other)
            )
            return self.__class__(  # type: ignore[call-arg]
                id=self.id,
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        msg = (
            'unsupported operand type(s) for +: "'
            f"{self.__class__.__name__}"
            f'" and "{other.__class__.__name__}"'
        )
        raise TypeError(msg)


def message_to_dict(message: BaseMessage) -> dict:
    """将 Message 转换为字典。

    参数:
        message: 要转换的 Message 对象。

    返回:
        消息的字典表示。该字典将包含一个带有消息类型的 "type" 键
        和一个带有消息数据的 "data" 键。
    """
    return {"type": message.type, "data": message.model_dump()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict]:
    """将一个 Message 序列转换为字典列表。

    参数:
        messages: 要转换的 Message 序列（作为 BaseMessage 对象）。

    返回:
        消息的字典列表。
    """
    return [message_to_dict(m) for m in messages]


def get_msg_title_repr(title: str, *, bold: bool = False) -> str:
    """获取消息的标题表示。

    参数:
        title: 标题。
        bold: 是否加粗标题。默认为 False。

    返回:
        标题的表示形式。
    """
    padded = " " + title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    if bold:
        padded = get_bolded_text(padded)
    return f"{sep}{padded}{second_sep}"
