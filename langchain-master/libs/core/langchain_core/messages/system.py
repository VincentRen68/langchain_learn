"""系统消息。"""

from typing import Any, Literal, Union

from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class SystemMessage(BaseMessage):
    """用于引导 AI 行为的消息。

    系统消息通常作为输入消息序列中的第一条传入。

    示例:

        .. code-block:: python

            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="你是一个乐于助人的助手！你的名字叫鲍勃。"),
                HumanMessage(content="你叫什么名字？"),
            ]

            # 定义一个聊天模型并用消息列表调用它
            print(model.invoke(messages))

    """

    type: Literal["system"] = "system"
    """消息的类型（用于反序列化）。默认为 "system"。"""

    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        """将 content 作为位置参数传入。

        参数:
               content: 消息的字符串内容。
               kwargs: 要传递给消息的其他字段。
        """
        super().__init__(content=content, **kwargs)


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """系统消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于反序列化）。
    默认为 "SystemMessageChunk"。"""
