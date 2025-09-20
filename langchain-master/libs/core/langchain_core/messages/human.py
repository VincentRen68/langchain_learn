"""人类消息。"""

from typing import Any, Literal, Union

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class HumanMessage(BaseMessage):
    """来自人类的消息。

    HumanMessage 是从人类传递给模型的消息。

    示例:

        .. code-block:: python

            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="你是一个乐于助人的助手！你的名字叫鲍勃。"),
                HumanMessage(content="你叫什么名字？"),
            ]

            # 实例化一个聊天模型并用消息列表调用它
            model = ...
            print(model.invoke(messages))

    """

    example: bool = False
    """用于表示此消息是一段示例对话的一部分。

    目前，大多数模型会忽略此字段。不建议使用。
    默认为 False。
    """

    type: Literal["human"] = "human"
    """消息的类型（用于反序列化）。默认为 "human"。"""

    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        """将 content 作为位置参数传入。

        参数:
            content: 消息的字符串内容。
            kwargs: 要传递给消息的其他字段。
        """
        super().__init__(content=content, **kwargs)


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """人类消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于反序列化）。
    默认为 "HumanMessageChunk"。"""
