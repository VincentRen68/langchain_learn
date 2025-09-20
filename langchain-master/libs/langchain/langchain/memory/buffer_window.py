from typing import Any, Union

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from typing_extensions import override

from langchain.memory.chat_memory import BaseChatMemory


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "请参阅迁移指南："
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationBufferWindowMemory(BaseChatMemory):
    """用于追踪对话的最近 k 轮。

    如果对话中的消息数量超过了要保留的最大消息数，
    最旧的消息将被丢弃。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    k: int = 5
    """要在缓冲区中存储的对话轮数。"""

    @property
    def buffer(self) -> Union[str, list[BaseMessage]]:
        """内存的字符串缓冲区。"""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """在 return_messages 为 False 的情况下，将缓冲区暴露为字符串。"""
        messages = self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """在 return_messages 为 True 的情况下，将缓冲区暴露为消息列表。"""
        return self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []

    @property
    def memory_variables(self) -> list[str]:
        """总是返回内存变量的列表。

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        return {self.memory_key: self.buffer}
