from typing import Any, Optional

from langchain_core._api import deprecated
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils import pre_init
from typing_extensions import override

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "请参阅迁移指南："
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationBufferMemory(BaseChatMemory):
    """一个基础的记忆实现，仅简单地存储对话历史。

    它将整个对话历史存储在内存中，不进行任何额外处理。

    请注意，在某些情况下，当对话历史过大以至于无法放入模型的上下文窗口时，
    可能需要进行额外处理。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> Any:
        """内存的字符串缓冲区。"""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    async def abuffer(self) -> Any:
        """内存的字符串缓冲区。"""
        return (
            await self.abuffer_as_messages()
            if self.return_messages
            else await self.abuffer_as_str()
        )

    def _buffer_as_str(self, messages: list[BaseMessage]) -> str:
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_str(self) -> str:
        """在 return_messages 为 True 的情况下，将缓冲区暴露为字符串。"""
        return self._buffer_as_str(self.chat_memory.messages)

    async def abuffer_as_str(self) -> str:
        """在 return_messages 为 True 的情况下，将缓冲区暴露为字符串。"""
        messages = await self.chat_memory.aget_messages()
        return self._buffer_as_str(messages)

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """在 return_messages 为 False 的情况下，将缓冲区暴露为消息列表。"""
        return self.chat_memory.messages

    async def abuffer_as_messages(self) -> list[BaseMessage]:
        """在 return_messages 为 False 的情况下，将缓冲区暴露为消息列表。"""
        return await self.chat_memory.aget_messages()

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

    @override
    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据给定的文本输入，返回键值对。"""
        buffer = await self.abuffer()
        return {self.memory_key: buffer}


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "请参阅迁移指南："
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationStringBufferMemory(BaseMemory):
    """一个基础的记忆实现，仅简单地存储对话历史。

    它将整个对话历史存储在内存中，不进行任何额外处理。

    与 ConversationBufferMemory 等效，但更侧重于基于字符串的对话，
    而不是为聊天模型设计。

    请注意，在某些情况下，当对话历史过大以至于无法放入模型的上下文窗口时，
    可能需要进行额外处理。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    """用于 AI 生成的响应的前缀。"""
    buffer: str = ""
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    memory_key: str = "history"  #: :meta private:

    @pre_init
    def validate_chains(cls, values: dict) -> dict:
        """验证 return_messages 不为 True。"""
        if values.get("return_messages", False):
            msg = "对于 ConversationStringBufferMemory，return_messages 必须为 False"
            raise ValueError(msg)
        return values

    @property
    def memory_variables(self) -> list[str]:
        """总是返回内存变量的列表。

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """返回历史缓冲区。"""
        return {self.memory_key: self.buffer}

    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """返回历史缓冲区。"""
        return self.load_memory_variables(inputs)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                msg = f"预期一个输出键，但得到了 {outputs.keys()}"
                raise ValueError(msg)
            output_key = next(iter(outputs.keys()))
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.buffer += f"\n{human}\n{ai}"

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        return self.save_context(inputs, outputs)

    def clear(self) -> None:
        """清除记忆内容。"""
        self.buffer = ""

    @override
    async def aclear(self) -> None:
        self.clear()
