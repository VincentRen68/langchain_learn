from typing import Any

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from typing_extensions import override

from langchain.memory.chat_memory import BaseChatMemory


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationTokenBufferMemory(BaseChatMemory):
    """带 Token 限制的对话聊天记忆。

    在对话中的总 Token 数不超过特定限制的约束下，
    仅保留对话中最新的消息。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000

    @property
    def buffer(self) -> Any:
        """内存的字符串缓冲区。"""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """在 return_messages 为 False 的情况下，将缓冲区暴露为字符串。"""
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """在 return_messages 为 True 的情况下，将缓冲区暴露为消息列表。"""
        return self.chat_memory.messages

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

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区并进行修剪。"""
        super().save_context(inputs, outputs)
        # 如果缓冲区超过最大 Token 限制，则进行修剪
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
