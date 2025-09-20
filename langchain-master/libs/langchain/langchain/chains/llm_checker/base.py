"""用于自我验证的问答链。"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from pydantic import ConfigDict, model_validator

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.prompt import (
    CHECK_ASSERTIONS_PROMPT,
    CREATE_DRAFT_ANSWER_PROMPT,
    LIST_ASSERTIONS_PROMPT,
    REVISED_ANSWER_PROMPT,
)
from langchain.chains.sequential import SequentialChain


def _load_question_to_checked_assertions_chain(
    llm: BaseLanguageModel,
    create_draft_answer_prompt: PromptTemplate,
    list_assertions_prompt: PromptTemplate,
    check_assertions_prompt: PromptTemplate,
    revised_answer_prompt: PromptTemplate,
) -> SequentialChain:
    create_draft_answer_chain = LLMChain(
        llm=llm,
        prompt=create_draft_answer_prompt,
        output_key="statement",
    )
    list_assertions_chain = LLMChain(
        llm=llm,
        prompt=list_assertions_prompt,
        output_key="assertions",
    )
    check_assertions_chain = LLMChain(
        llm=llm,
        prompt=check_assertions_prompt,
        output_key="checked_assertions",
    )
    revised_answer_chain = LLMChain(
        llm=llm,
        prompt=revised_answer_prompt,
        output_key="revised_statement",
    )
    chains = [
        create_draft_answer_chain,
        list_assertions_chain,
        check_assertions_chain,
        revised_answer_chain,
    ]
    return SequentialChain(
        chains=chains,
        input_variables=["question"],
        output_variables=["revised_statement"],
        verbose=True,
    )


@deprecated(
    since="0.2.13",
    message=(
        "请参阅 LangGraph 指南了解各种自我反思和纠正策略，适用于问答和其他任务: "
        "https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/"
    ),
    removal="1.0",
)
class LLMCheckerChain(Chain):
    """用于自我验证的问答链。

    示例:
        .. code-block:: python

            from langchain_community.llms import OpenAI
            from langchain.chains import LLMCheckerChain

            llm = OpenAI(temperature=0.7)
            checker_chain = LLMCheckerChain.from_llm(llm)

    """

    question_to_checked_assertions_chain: SequentialChain

    llm: Optional[BaseLanguageModel] = None
    """[已弃用] 要使用的 LLM 包装器。"""
    create_draft_answer_prompt: PromptTemplate = CREATE_DRAFT_ANSWER_PROMPT
    """[已弃用]"""
    list_assertions_prompt: PromptTemplate = LIST_ASSERTIONS_PROMPT
    """[已弃用]"""
    check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT
    """[已弃用]"""
    revised_answer_prompt: PromptTemplate = REVISED_ANSWER_PROMPT
    """[已弃用] 查询文档时使用的提示。"""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _raise_deprecation(cls, values: dict) -> Any:
        if "llm" in values:
            warnings.warn(
                "直接使用 llm 实例化 LLMCheckerChain 已被弃用。"
                "请使用 question_to_checked_assertions_chain 实例化"
                "或使用 from_llm 类方法。",
                stacklevel=5,
            )
            if (
                "question_to_checked_assertions_chain" not in values
                and values["llm"] is not None
            ):
                question_to_checked_assertions_chain = (
                    _load_question_to_checked_assertions_chain(
                        values["llm"],
                        values.get(
                            "create_draft_answer_prompt",
                            CREATE_DRAFT_ANSWER_PROMPT,
                        ),
                        values.get("list_assertions_prompt", LIST_ASSERTIONS_PROMPT),
                        values.get("check_assertions_prompt", CHECK_ASSERTIONS_PROMPT),
                        values.get("revised_answer_prompt", REVISED_ANSWER_PROMPT),
                    )
                )
                values["question_to_checked_assertions_chain"] = (
                    question_to_checked_assertions_chain
                )
        return values

    @property
    def input_keys(self) -> list[str]:
        """返回单个输入键。

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """返回单个输出键。

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        output = self.question_to_checked_assertions_chain(
            {"question": question},
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: output["revised_statement"]}

    @property
    def _chain_type(self) -> str:
        return "llm_checker_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        create_draft_answer_prompt: PromptTemplate = CREATE_DRAFT_ANSWER_PROMPT,
        list_assertions_prompt: PromptTemplate = LIST_ASSERTIONS_PROMPT,
        check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT,
        revised_answer_prompt: PromptTemplate = REVISED_ANSWER_PROMPT,
        **kwargs: Any,
    ) -> LLMCheckerChain:
        """从语言模型创建 LLMCheckerChain。

        参数:
            llm: 语言模型
            create_draft_answer_prompt: 创建草稿答案的提示
            list_assertions_prompt: 列出断言的提示
            check_assertions_prompt: 检查断言的提示
            revised_answer_prompt: 修订答案的提示
            **kwargs: 额外参数
        """
        question_to_checked_assertions_chain = (
            _load_question_to_checked_assertions_chain(
                llm,
                create_draft_answer_prompt,
                list_assertions_prompt,
                check_assertions_prompt,
                revised_answer_prompt,
            )
        )
        return cls(
            question_to_checked_assertions_chain=question_to_checked_assertions_chain,
            **kwargs,
        )
