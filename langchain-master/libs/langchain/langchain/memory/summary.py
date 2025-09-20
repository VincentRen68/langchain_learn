from __future__ import annotations

from typing import Any

from langchain_core._api import deprecated
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from langchain_core.utils import pre_init
from pydantic import BaseModel
from typing_extensions import override

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import SUMMARY_PROMPT


@deprecated(
    since="0.2.12",
    removal="1.0",
    message=(
        "请参阅此处了解如何整合对话历史摘要："
        "https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/"
    ),
)
class SummarizerMixin(BaseModel):
    """用于摘要器的 Mixin。"""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    summary_message_cls: type[BaseMessage] = SystemMessage

    def predict_new_summary(
        self,
        messages: list[BaseMessage],
        existing_summary: str,
    ) -> str:
        """根据消息和现有摘要预测新的摘要。

        参数:
            messages: 需要进行摘要的消息列表。
            existing_summary: 用于在其上构建的现有摘要。

        返回:
            一个新的摘要字符串。
        """
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)

    async def apredict_new_summary(
        self,
        messages: list[BaseMessage],
        existing_summary: str,
    ) -> str:
        """根据消息和现有摘要异步预测新的摘要。

        参数:
            messages: 需要进行摘要的消息列表。
            existing_summary: 用于在其上构建的现有摘要。

        返回:
            一个新的摘要字符串。
        """
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return await chain.apredict(summary=existing_summary, new_lines=new_lines)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "请参阅迁移指南："
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationSummaryMemory(BaseChatMemory, SummarizerMixin):
    """持续总结对话历史。

    摘要在每一轮对话后更新。
    该实现返回对话历史的摘要，
    可用于为模型提供上下文。
    """

    buffer: str = ""
    memory_key: str = "history"  #: :meta private:

    @classmethod
    def from_messages(
        cls,
        llm: BaseLanguageModel,
        chat_memory: BaseChatMessageHistory,
        *,
        summarize_step: int = 2,
        **kwargs: Any,
    ) -> ConversationSummaryMemory:
        """从消息列表创建 ConversationSummaryMemory。

        参数:
            llm: 用于摘要的语言模型。
            chat_memory: 需要摘要的聊天历史。
            summarize_step: 一次摘要的消息数量。
            **kwargs: 传递给类的其他关键字参数。

        返回:
            一个包含已摘要历史的 ConversationSummaryMemory 实例。
        """
        obj = cls(llm=llm, chat_memory=chat_memory, **kwargs)
        for i in range(0, len(obj.chat_memory.messages), summarize_step):
            obj.buffer = obj.predict_new_summary(
                obj.chat_memory.messages[i : i + summarize_step],
                obj.buffer,
            )
        return obj

    @property
    def memory_variables(self) -> list[str]:
        """总是返回内存变量的列表。

        :meta private:
        """
        return [self.memory_key]

    @override
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史缓冲区。"""
        if self.return_messages:
            buffer: Any = [self.summary_message_cls(content=self.buffer)]
        else:
            buffer = self.buffer
        return {self.memory_key: buffer}

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
        self.buffer = self.predict_new_summary(
            self.chat_memory.messages[-2:],
            self.buffer,
        )

    def clear(self) -> None:
        """清除记忆内容。"""
        super().clear()
        self.buffer = ""
