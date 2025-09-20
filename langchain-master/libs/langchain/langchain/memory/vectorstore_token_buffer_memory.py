"""一个将旧消息存储在向量存储中的对话记忆缓冲区类。

该类实现了一种对话记忆，其中消息被存储在一个内存缓冲区中，
直到达到指定的 Token 限制。当超出限制时，较旧的消息将被保存到向量存储支持的数据库中。
向量存储可以在不同会话之间持久化。
"""

import warnings
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field, PrivateAttr

from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_HISTORY_TEMPLATE = """
当前日期和时间: {current_time}。

先前对话中可能相关且带有时间戳的摘录 (如果不相关，则无需使用):
{previous_history}

"""

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %Z"


class ConversationVectorStoreTokenBufferMemory(ConversationTokenBufferMemory):
    """带有 Token 限制和向量数据库支持的对话聊天记忆。

    load_memory_variables() 将返回一个包含键“history”的字典。
    它包含从向量存储中检索到的背景信息，以及当前对话的最近几行。

    为了帮助 LLM 理解存储在向量存储中的对话部分，
    每次交互都会被加上时间戳，并且当前日期和时间也会在历史记录中提供。
    这样做的一个副作用是，LLM 将能够访问当前的日期和时间。

    初始化参数:

    该类接受 ConversationTokenBufferMemory 的所有初始化参数，例如 `llm`。
    此外，它还接受以下附加参数：

        retriever: (必需) 一个 VectorStoreRetriever 对象，用作向量后备存储。

        split_chunk_size: (可选, 默认为 1000) AI 生成的长消息的 Token 块拆分大小。

        previous_history_template: (可选) 用于格式化提示历史内容的模板。


    使用 ChromaDB 的示例:

    .. code-block:: python

        from langchain.memory.token_buffer_vectorstore_memory import (
            ConversationVectorStoreTokenBufferMemory,
        )
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceInstructEmbeddings
        from langchain_openai import OpenAI

        embedder = HuggingFaceInstructEmbeddings(
            query_instruction="Represent the query for retrieval: "
        )
        chroma = Chroma(
            collection_name="demo",
            embedding_function=embedder,
            collection_metadata={"hnsw:space": "cosine"},
        )

        retriever = chroma.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.75,
            },
        )

        conversation_memory = ConversationVectorStoreTokenBufferMemory(
            return_messages=True,
            llm=OpenAI(),
            retriever=retriever,
            max_token_limit=1000,
        )

        conversation_memory.save_context(
            {"Human": "Hi there"}, {"AI": "Nice to meet you!"}
        )
        conversation_memory.save_context(
            {"Human": "Nice day isn't it?"}, {"AI": "I love Wednesdays."}
        )
        conversation_memory.load_memory_variables({"input": "What time is it?"})

    """

    retriever: VectorStoreRetriever = Field(exclude=True)
    memory_key: str = "history"
    previous_history_template: str = DEFAULT_HISTORY_TEMPLATE
    split_chunk_size: int = 1000

    _memory_retriever: Optional[VectorStoreRetrieverMemory] = PrivateAttr(default=None)
    _timestamps: list[datetime] = PrivateAttr(default_factory=list)

    @property
    def memory_retriever(self) -> VectorStoreRetrieverMemory:
        """从传入的 retriever 对象返回一个记忆检索器。"""
        if self._memory_retriever is not None:
            return self._memory_retriever
        self._memory_retriever = VectorStoreRetrieverMemory(retriever=self.retriever)
        return self._memory_retriever

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回历史和记忆缓冲区。"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                memory_variables = self.memory_retriever.load_memory_variables(inputs)
            previous_history = memory_variables[self.memory_retriever.memory_key]
        except AssertionError:  # 当数据库为空时发生
            previous_history = ""
        current_history = super().load_memory_variables(inputs)
        template = SystemMessagePromptTemplate.from_template(
            self.previous_history_template,
        )
        messages = [
            template.format(
                previous_history=previous_history,
                current_time=datetime.now().astimezone().strftime(TIMESTAMP_FORMAT),
            ),
        ]
        messages.extend(current_history[self.memory_key])
        return {self.memory_key: messages}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区并进行修剪。"""
        BaseChatMemory.save_context(self, inputs, outputs)
        self._timestamps.append(datetime.now().astimezone())
        # 如果缓冲区超过最大 Token 限制，则进行修剪
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            while curr_buffer_length > self.max_token_limit:
                self._pop_and_store_interaction(buffer)
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)

    def save_remainder(self) -> None:
        """将对话缓冲区的剩余部分保存到向量存储中。

        如果已将向量存储设置为持久化，此功能将非常有用，
        在这种情况下，可以在会话结束前调用此方法以存储对话的剩余部分。
        """
        buffer = self.chat_memory.messages
        while len(buffer) > 0:
            self._pop_and_store_interaction(buffer)

    def _pop_and_store_interaction(self, buffer: list[BaseMessage]) -> None:
        input_ = buffer.pop(0)
        output = buffer.pop(0)
        timestamp = self._timestamps.pop(0).strftime(TIMESTAMP_FORMAT)
        # 将 AI 输出拆分为更小的块，以避免创建会溢出上下文窗口的文档
        ai_chunks = self._split_long_ai_text(str(output.content))
        for index, chunk in enumerate(ai_chunks):
            self.memory_retriever.save_context(
                {"Human": f"<{timestamp}/00> {input_.content!s}"},
                {"AI": f"<{timestamp}/{index:02}> {chunk}"},
            )

    def _split_long_ai_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.split_chunk_size)
        return [chunk.page_content for chunk in splitter.create_documents([text])]
