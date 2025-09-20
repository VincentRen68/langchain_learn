"""仅格式化提示并调用 LLM 的链。"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Optional, Union, cast

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.language_models import (
    BaseLanguageModel,
    LanguageModelInput,
)
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseLLMOutputParser, StrOutputParser
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableBranch,
    RunnableWithFallbacks,
)
from langchain_core.runnables.configurable import DynamicRunnable
from langchain_core.utils.input import get_colored_text
from pydantic import ConfigDict, Field
from typing_extensions import override

from langchain.chains.base import Chain


@deprecated(
    since="0.1.17",
    alternative="RunnableSequence, e.g., `prompt | llm`",
    removal="1.0",
)
class LLMChain(Chain):
    """针对 LLM 运行查询的链。

    此类已弃用。请参阅下面使用 LangChain 可运行组件的示例实现：

        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import OpenAI

            prompt_template = "给我讲一个{adjective}的笑话"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = OpenAI()
            chain = prompt | llm | StrOutputParser()

            chain.invoke("这里是你的形容词")

    示例:
        .. code-block:: python

            from langchain.chains import LLMChain
            from langchain_community.llms import OpenAI
            from langchain_core.prompts import PromptTemplate

            prompt_template = "给我讲一个{adjective}的笑话"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)

    """

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    prompt: BasePromptTemplate
    """要使用的提示对象。"""
    llm: Union[
        Runnable[LanguageModelInput, str],
        Runnable[LanguageModelInput, BaseMessage],
    ]
    """要调用的语言模型。"""
    output_key: str = "text"  #: :meta private:
    output_parser: BaseLLMOutputParser = Field(default_factory=StrOutputParser)
    """要使用的输出解析器。
    默认为一个获取最可能字符串但不以其他方式更改它的解析器。"""
    return_final_only: bool = True
    """是否仅返回最终解析结果。默认为 True。
    如果为 False，将返回大量关于生成的额外信息。"""
    llm_kwargs: dict = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @property
    def input_keys(self) -> list[str]:
        """将是提示所期望的任何键。

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> list[str]:
        """将始终返回文本键。

        :meta private:
        """
        if self.return_final_only:
            return [self.output_key]
        return [self.output_key, "full_generation"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        response = self.generate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def generate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """从输入生成 LLM 结果。"""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            return self.llm.generate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        results = self.llm.bind(stop=stop, **self.llm_kwargs).batch(
            cast("list", prompts),
            {"callbacks": callbacks},
        )
        generations: list[list[Generation]] = []
        for res in results:
            if isinstance(res, BaseMessage):
                generations.append([ChatGeneration(message=res)])
            else:
                generations.append([Generation(text=res)])
        return LLMResult(generations=generations)

    async def agenerate(
        self,
        input_list: list[dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """从输入生成 LLM 结果。"""
        prompts, stop = await self.aprep_prompts(input_list, run_manager=run_manager)
        callbacks = run_manager.get_child() if run_manager else None
        if isinstance(self.llm, BaseLanguageModel):
            return await self.llm.agenerate_prompt(
                prompts,
                stop,
                callbacks=callbacks,
                **self.llm_kwargs,
            )
        results = await self.llm.bind(stop=stop, **self.llm_kwargs).abatch(
            cast("list", prompts),
            {"callbacks": callbacks},
        )
        generations: list[list[Generation]] = []
        for res in results:
            if isinstance(res, BaseMessage):
                generations.append([ChatGeneration(message=res)])
            else:
                generations.append([Generation(text=res)])
        return LLMResult(generations=generations)

    def prep_prompts(
        self,
        input_list: list[dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> tuple[list[PromptValue], Optional[list[str]]]:
        """从输入准备提示。"""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "格式化后的提示:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                msg = "如果 `stop` 出现在任何输入中，则应出现在所有输入中。"
                raise ValueError(msg)
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(
        self,
        input_list: list[dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> tuple[list[PromptValue], Optional[list[str]]]:
        """从输入准备提示。"""
        stop = None
        if len(input_list) == 0:
            return [], stop
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "格式化后的提示:\n" + _colored_text
            if run_manager:
                await run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                msg = "如果 `stop` 出现在任何输入中，则应出现在所有输入中。"
                raise ValueError(msg)
            prompts.append(prompt)
        return prompts, stop

    def apply(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> list[dict[str, str]]:
        """利用 LLM 生成方法以提高速度。"""
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
        )
        run_manager = callback_manager.on_chain_start(
            None,
            {"input_list": input_list},
            name=self.get_name(),
        )
        try:
            response = self.generate(input_list, run_manager=run_manager)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        outputs = self.create_outputs(response)
        run_manager.on_chain_end({"outputs": outputs})
        return outputs

    async def aapply(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> list[dict[str, str]]:
        """利用 LLM 生成方法以提高速度。"""
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
        )
        run_manager = await callback_manager.on_chain_start(
            None,
            {"input_list": input_list},
            name=self.get_name(),
        )
        try:
            response = await self.agenerate(input_list, run_manager=run_manager)
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        outputs = self.create_outputs(response)
        await run_manager.on_chain_end({"outputs": outputs})
        return outputs

    @property
    def _run_output_key(self) -> str:
        return self.output_key

    def create_outputs(self, llm_result: LLMResult) -> list[dict[str, Any]]:
        """从响应创建输出。"""
        result = [
            # 获取顶部生成字符串的文本。
            {
                self.output_key: self.output_parser.parse_result(generation),
                "full_generation": generation,
            }
            for generation in llm_result.generations
        ]
        if self.return_final_only:
            result = [{self.output_key: r[self.output_key]} for r in result]
        return result

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        response = await self.agenerate([inputs], run_manager=run_manager)
        return self.create_outputs(response)[0]

    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """使用 kwargs 格式化提示并传递给 LLM。

        参数:
            callbacks: 传递给 LLMChain 的回调。
            **kwargs: 传递给提示模板的键。

        返回:
            来自 LLM 的补全。

        示例:
            .. code-block:: python

                completion = llm.predict(adjective="funny")

        """
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """使用 kwargs 格式化提示并传递给 LLM。

        参数:
            callbacks: 传递给 LLMChain 的回调。
            **kwargs: 传递给提示模板的键。

        返回:
            来自 LLM 的补全。

        示例:
            .. code-block:: python

                completion = llm.predict(adjective="funny")

        """
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]

    def predict_and_parse(
        self,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[str, list[str], dict[str, Any]]:
        """调用 predict 然后解析结果。"""
        warnings.warn(
            "predict_and_parse 方法已弃用，"
            "请直接将输出解析器传递给 LLMChain。",
            stacklevel=2,
        )
        result = self.predict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        return result

    async def apredict_and_parse(
        self,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[str, list[str], dict[str, str]]:
        """调用 apredict 然后解析结果。"""
        warnings.warn(
            "apredict_and_parse 方法已弃用，"
            "请直接将输出解析器传递给 LLMChain。",
            stacklevel=2,
        )
        result = await self.apredict(callbacks=callbacks, **kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        return result

    def apply_and_parse(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> Sequence[Union[str, list[str], dict[str, str]]]:
        """调用 apply 然后解析结果。"""
        warnings.warn(
            "apply_and_parse 方法已弃用，"
            "请直接将输出解析器传递给 LLMChain。",
            stacklevel=2,
        )
        result = self.apply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    def _parse_generation(
        self,
        generation: list[dict[str, str]],
    ) -> Sequence[Union[str, list[str], dict[str, str]]]:
        if self.prompt.output_parser is not None:
            return [
                self.prompt.output_parser.parse(res[self.output_key])
                for res in generation
            ]
        return generation

    async def aapply_and_parse(
        self,
        input_list: list[dict[str, Any]],
        callbacks: Callbacks = None,
    ) -> Sequence[Union[str, list[str], dict[str, str]]]:
        """调用 apply 然后解析结果。"""
        warnings.warn(
            "aapply_and_parse 方法已弃用，"
            "请直接将输出解析器传递给 LLMChain。",
            stacklevel=2,
        )
        result = await self.aapply(input_list, callbacks=callbacks)
        return self._parse_generation(result)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> LLMChain:
        """从 LLM 和模板创建 LLMChain。"""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)

    def _get_num_tokens(self, text: str) -> int:
        return _get_language_model(self.llm).get_num_tokens(text)


def _get_language_model(llm_like: Runnable) -> BaseLanguageModel:
    if isinstance(llm_like, BaseLanguageModel):
        return llm_like
    if isinstance(llm_like, RunnableBinding):
        return _get_language_model(llm_like.bound)
    if isinstance(llm_like, RunnableWithFallbacks):
        return _get_language_model(llm_like.runnable)
    if isinstance(llm_like, (RunnableBranch, DynamicRunnable)):
        return _get_language_model(llm_like.default)
    msg = (
        f"无法从类型为 {type(llm_like)} 的 llm_like 对象中提取 BaseLanguageModel"
    )
    raise ValueError(msg)
