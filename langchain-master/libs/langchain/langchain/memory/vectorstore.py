"""由向量存储支持的记忆对象的类。"""

from collections.abc import Sequence
from typing import Any, Optional, Union

from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.memory import BaseMemory
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field

from langchain.memory.utils import get_prompt_input_key


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "请参阅迁移指南："
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class VectorStoreRetrieverMemory(BaseMemory):
    """向量存储检索器记忆。

    将对话历史存储在向量存储中，并根据输入检索过去对话的相关部分。
    """

    retriever: VectorStoreRetriever = Field(exclude=True)
    """用于连接的 VectorStoreRetriever 对象。"""

    memory_key: str = "history"  #: :meta private:
    """用于在 load_memory_variables 的结果中定位记忆的键名。"""

    input_key: Optional[str] = None
    """用于索引 load_memory_variables 输入的键名。"""

    return_docs: bool = False
    """是否直接返回查询数据库的结果。"""

    exclude_input_keys: Sequence[str] = Field(default_factory=tuple)
    """在构建文档时，除了记忆键之外要排除的输入键。"""

    @property
    def memory_variables(self) -> list[str]:
        """从 load_memory_variables 方法发出的键列表。"""
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: dict[str, Any]) -> str:
        """获取提示的输入键。"""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def _documents_to_memory_variables(
        self,
        docs: list[Document],
    ) -> dict[str, Union[list[Document], str]]:
        result: Union[list[Document], str]
        if not self.return_docs:
            result = "\n".join([doc.page_content for doc in docs])
        else:
            result = docs
        return {self.memory_key: result}

    def load_memory_variables(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Union[list[Document], str]]:
        """返回历史缓冲区。"""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.retriever.invoke(query)
        return self._documents_to_memory_variables(docs)

    async def aload_memory_variables(
        self,
        inputs: dict[str, Any],
    ) -> dict[str, Union[list[Document], str]]:
        """返回历史缓冲区。"""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = await self.retriever.ainvoke(query)
        return self._documents_to_memory_variables(docs)

    def _form_documents(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> list[Document]:
        """将本次对话的上下文格式化为缓冲区。"""
        # 每个文档应只包含当前轮次，不包含聊天历史
        exclude = set(self.exclude_input_keys)
        exclude.add(self.memory_key)
        filtered_inputs = {k: v for k, v in inputs.items() if k not in exclude}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        documents = self._form_documents(inputs, outputs)
        self.retriever.add_documents(documents)

    async def asave_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, str],
    ) -> None:
        """将本次对话的上下文保存到缓冲区。"""
        documents = self._form_documents(inputs, outputs)
        await self.retriever.aadd_documents(documents)

    def clear(self) -> None:
        """无需清除任何内容。"""

    async def aclear(self) -> None:
        """无需清除任何内容。"""
