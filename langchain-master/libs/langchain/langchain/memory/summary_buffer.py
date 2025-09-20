from typing import Any, Union

from langchain_core._api import deprecated
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.utils import pre_init
from typing_extensions import override

from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.summary import SummarizerMixin


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "请参阅迁移指南："
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin):
    """带有摘要器的缓冲区，用于存储对话记忆。

    在对话的总 Token 数不超过特定限制的约束下，
    提供对话的滚动摘要以及最新的消息。
    """

    max_token_limit: int = 2000
    moving_summary_buffer: str = ""
    memory_key: str = "history"

    @property
    def buffer(self) -> Union[str, list[BaseMessage]]:
        """内存的字符串缓冲区。"""
        return self.load_memory_variables({})[self.memory_key]

    async def abuffer(self) -> Union[str, list[BaseMessage]]:
        """异步内存缓冲区。"""
        memory_variables = await self.aload_memory_variables({})
        return memory_variables[self.memory_key]

    @property
    def memory_variables(self) -> list[str]:
        """总是返回内存变量的列表。

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        buffer = self.chat_memory.messages
        if self.moving_summary_buffer != "":
            first_messages: list[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer),
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    @override
    async def aload_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据给定的文本输入，异步返回键值对。"""
        buffer = await self.chat_memory.aget_messages()
        if self.moving_summary_buffer != "":
            first_messages: list[BaseMessage] = [
                self.summary_message_cls(content=self.moving_summary_buffer),
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    @pre_init
    def validate_prompt_input_variables(cls, values: dict) -> dict:
        """验证提示输入变量是否一致。"""
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            msg = (
                "获得了意外的提示输入变量。提示预期 "
                f"{prompt_variables}，但应为 {expected_keys}。"
            )
            raise ValueError(msg)
        return values

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        super().save_context(inputs, outputs)
        self.prune()

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """将本次对话的上下文异步保存到缓冲区。"""
        await super().asave_context(inputs, outputs)
        await self.aprune()

    def prune(self) -> None:
        """如果缓冲区超过最大 Token 限制，则进行修剪。"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory,
                self.moving_summary_buffer,
            )

    async def aprune(self) -> None:
        """如果缓冲区超过最大 Token 限制，则异步进行修剪。"""
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = await self.apredict_new_summary(
                pruned_memory,
                self.moving_summary_buffer,
            )

    def clear(self) -> None:
        """清除记忆内容。"""
        super().clear()
        self.moving_summary_buffer = ""

    async def aclear(self) -> None:
        """异步清除记忆内容。"""
        await super().aclear()
        self.moving_summary_buffer = ""
