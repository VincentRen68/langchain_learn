"""接收输入并产生动作和动作输入的链。"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import logging
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

import yaml
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, override

from langchain._api.deprecation import AGENT_DEPRECATION_WARNING
from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.agents.agent_types import AgentType
from langchain.agents.tools import InvalidTool
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.utilities.asyncio import asyncio_timeout

logger = logging.getLogger(__name__)


class BaseSingleActionAgent(BaseModel):
    """基础单动作代理类。"""

    @property
    def return_values(self) -> list[str]:
        """代理的返回值。"""
        return ["output"]

    def get_allowed_tools(self) -> Optional[list[str]]:
        """获取允许的工具。"""
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """根据输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """异步地根据输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """返回输入键。

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """当代理因达到最大迭代次数而停止时返回响应。

        参数:
            early_stopping_method: 用于提前停止的方法。
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。

        返回:
            AgentFinish: 代理完成对象。

        异常:
            ValueError: 如果不支持 `early_stopping_method`。
        """
        if early_stopping_method == "force":
            # `force` 只返回一个常量字符串
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        msg = f"获取了不支持的 early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> BaseSingleActionAgent:
        """从 LLM 和工具构建代理。

        参数:
            llm: 要使用的语言模型。
            tools: 要使用的工具。
            callback_manager: 要使用的回调管理器。
            kwargs: 额外参数。

        返回:
            BaseSingleActionAgent: 代理对象。
        """
        raise NotImplementedError

    @property
    def _agent_type(self) -> str:
        """返回代理类型的标识符。"""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回代理的字典表示。

        返回:
            Dict: 代理的字典表示。
        """
        _dict = super().model_dump()
        try:
            _type = self._agent_type
        except NotImplementedError:
            _type = None
        if isinstance(_type, AgentType):
            _dict["_type"] = str(_type.value)
        elif _type is not None:
            _dict["_type"] = _type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """保存代理。

        参数:
            file_path: 保存代理的文件路径。

        示例:
        .. code-block:: python

            # 如果使用代理执行器
            agent.agent.save(file_path="path/agent.yaml")

        """
        # 将文件转换为 Path 对象。
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # 获取要保存的字典
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"代理 {self} 不支持保存"
            raise NotImplementedError(msg)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} 必须是 json 或 yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志记录参数。"""
        return {}


class BaseMultiActionAgent(BaseModel):
    """基础多动作代理类。"""

    @property
    def return_values(self) -> list[str]:
        """代理的返回值。"""
        return ["output"]

    def get_allowed_tools(self) -> Optional[list[str]]:
        """获取允许的工具。

        返回:
            Optional[List[str]]: 允许的工具。
        """
        return None

    @abstractmethod
    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[list[AgentAction], AgentFinish]:
        """根据输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[list[AgentAction], AgentFinish]:
        """异步地根据输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        """返回输入键。

        :meta private:
        """

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],  # noqa: ARG002
        **_: Any,
    ) -> AgentFinish:
        """当代理因达到最大迭代次数而停止时返回响应。

        参数:
            early_stopping_method: 用于提前停止的方法。
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。

        返回:
            AgentFinish: 代理完成对象。

        异常:
            ValueError: 如果不支持 `early_stopping_method`。
        """
        if early_stopping_method == "force":
            # `force` 只返回一个常量字符串
            return AgentFinish({"output": "Agent stopped due to max iterations."}, "")
        msg = f"获取了不支持的 early_stopping_method `{early_stopping_method}`"
        raise ValueError(msg)

    @property
    def _agent_type(self) -> str:
        """返回代理类型的标识符。"""
        raise NotImplementedError

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回代理的字典表示。"""
        _dict = super().model_dump()
        with contextlib.suppress(NotImplementedError):
            _dict["_type"] = str(self._agent_type)
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """保存代理。

        参数:
            file_path: 保存代理的文件路径。

        异常:
            NotImplementedError: 如果代理不支持保存。
            ValueError: 如果 file_path 不是 json 或 yaml。

        示例:
        .. code-block:: python

            # 如果使用代理执行器
            agent.agent.save(file_path="path/agent.yaml")

        """
        # 将文件转换为 Path 对象。
        save_path = Path(file_path) if isinstance(file_path, str) else file_path

        # 获取要保存的字典
        agent_dict = self.dict()
        if "_type" not in agent_dict:
            msg = f"代理 {self} 不支持保存。"
            raise NotImplementedError(msg)

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".json":
            with save_path.open("w") as f:
                json.dump(agent_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w") as f:
                yaml.dump(agent_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} 必须是 json 或 yaml"
            raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志记录参数。"""
        return {}


class AgentOutputParser(BaseOutputParser[Union[AgentAction, AgentFinish]]):
    """用于将代理输出解析为代理动作/完成的基础类。"""

    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """将文本解析为代理动作/完成。"""


class MultiActionAgentOutputParser(
    BaseOutputParser[Union[list[AgentAction], AgentFinish]],
):
    """用于将代理输出解析为代理动作/完成的基础类。

    这用于可以返回多个动作的代理。
    """

    @abstractmethod
    def parse(self, text: str) -> Union[list[AgentAction], AgentFinish]:
        """将文本解析为代理动作/完成。

        参数:
            text: 要解析的文本。

        返回:
            Union[List[AgentAction], AgentFinish]:
                代理动作列表或代理完成。
        """


class RunnableAgent(BaseSingleActionAgent):
    """由 Runnable 驱动的代理。"""

    runnable: Runnable[dict, Union[AgentAction, AgentFinish]]
    """用于调用以获取代理动作的 Runnable。"""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """是否从 runnable 流式传输。

    如果为 True，则以流式方式调用底层 LLM，以便在使用带有代理执行器的 stream_log 时可以访问单个 LLM 令牌。
    如果为 False，则以非流式方式调用 LLM，并且单个 LLM 令牌在 stream_log 中将不可用。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """代理的返回值。"""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。"""
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """根据过去的历史和当前输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # 使用流式传输以确保底层 LLM 以流式方式调用，
            # 以便在使用带有代理执行器的 stream_log 时可以访问单个 LLM 令牌。
            # 因为 plan 的响应不是生成器，所以我们需要将输出累积到最终输出中并返回。
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[
        AgentAction,
        AgentFinish,
    ]:
        """异步地根据过去的历史和当前输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # 使用流式传输以确保底层 LLM 以流式方式调用，
            # 以便在使用带有代理执行器的 stream_log 时可以访问单个 LLM 令牌。
            # 因为 plan 的响应不是生成器，所以我们需要将输出累积到最终输出中并返回。
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )
        return final_output


class RunnableMultiActionAgent(BaseMultiActionAgent):
    """由 Runnable 驱动的代理。"""

    runnable: Runnable[dict, Union[list[AgentAction], AgentFinish]]
    """用于调用以获取代理动作的 Runnable。"""
    input_keys_arg: list[str] = []
    return_keys_arg: list[str] = []
    stream_runnable: bool = True
    """是否从 runnable 流式传输。

    如果为 True，则以流式方式调用底层 LLM，以便在使用带有代理执行器的 stream_log 时可以访问单个 LLM 令牌。
    如果为 False，则以非流式方式调用 LLM，并且单个 LLM 令牌在 stream_log 中将不可用。
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def return_values(self) -> list[str]:
        """代理的返回值。"""
        return self.return_keys_arg

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。

        返回:
            输入键列表。
        """
        return self.input_keys_arg

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[
        list[AgentAction],
        AgentFinish,
    ]:
        """根据过去的历史和当前输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # 使用流式传输以确保底层 LLM 以流式方式调用，
            # 以便在使用带有代理执行器的 stream_log 时可以访问单个 LLM 令牌。
            # 因为 plan 的响应不是生成器，所以我们需要将输出累积到最终输出中并返回。
            for chunk in self.runnable.stream(inputs, config={"callbacks": callbacks}):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = self.runnable.invoke(inputs, config={"callbacks": callbacks})

        return final_output

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[
        list[AgentAction],
        AgentFinish,
    ]:
        """异步地根据过去的历史和当前输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        inputs = {**kwargs, "intermediate_steps": intermediate_steps}
        final_output: Any = None
        if self.stream_runnable:
            # 使用流式传输以确保底层 LLM 以流式方式调用，
            # 以便在使用带有代理执行器的 stream_log 时可以访问单个 LLM 令牌。
            # 因为 plan 的响应不是生成器，所以我们需要将输出累积到最终输出中并返回。
            async for chunk in self.runnable.astream(
                inputs,
                config={"callbacks": callbacks},
            ):
                if final_output is None:
                    final_output = chunk
                else:
                    final_output += chunk
        else:
            final_output = await self.runnable.ainvoke(
                inputs,
                config={"callbacks": callbacks},
            )

        return final_output


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class LLMSingleActionAgent(BaseSingleActionAgent):
    """单动作代理的基础类。"""

    llm_chain: LLMChain
    """用于代理的 LLMChain。"""
    output_parser: AgentOutputParser
    """用于代理的输出解析器。"""
    stop: list[str]
    """停止时使用的字符串列表。"""

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。

        返回:
            输入键列表。
        """
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回代理的字典表示。"""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """根据输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """异步地根据输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志记录参数。"""
        return {
            "llm_prefix": "",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }


@deprecated(
    "0.1.0",
    message=AGENT_DEPRECATION_WARNING,
    removal="1.0",
)
class Agent(BaseSingleActionAgent):
    """调用语言模型并决定动作的代理。

    这由 LLMChain 驱动。LLMChain 中的提示必须包含一个名为“agent_scratchpad”的变量，
    代理可以在其中放置其中间工作。
    """

    llm_chain: LLMChain
    """用于代理的 LLMChain。"""
    output_parser: AgentOutputParser
    """用于代理的输出解析器。"""
    allowed_tools: Optional[list[str]] = None
    """代理允许的工具。如果为 None，则允许所有工具。"""

    @override
    def dict(self, **kwargs: Any) -> builtins.dict:
        """返回代理的字典表示。"""
        _dict = super().dict()
        del _dict["output_parser"]
        return _dict

    def get_allowed_tools(self) -> Optional[list[str]]:
        """获取允许的工具。"""
        return self.allowed_tools

    @property
    def return_values(self) -> list[str]:
        """代理的返回值。"""
        return ["output"]

    @property
    def _stop(self) -> list[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> Union[str, list[BaseMessage]]:
        """构建便笺簿，让代理能够继续其思考过程。"""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
        return thoughts

    def plan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """根据输入，决定做什么。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            callbacks: 要运行的回调。
            **kwargs: 用户输入。

        返回:
            指定使用哪个工具的动作。
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = self.llm_chain.predict(callbacks=callbacks, **full_inputs)
        return self.output_parser.parse(full_output)

    async def aplan(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Async given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        full_output = await self.llm_chain.apredict(callbacks=callbacks, **full_inputs)
        return await self.output_parser.aparse(full_output)

    def get_full_inputs(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> builtins.dict[str, Any]:
        """从中间步骤为 LLMChain 创建完整输入。

        参数:
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            **kwargs: 用户输入。

        返回:
            Dict[str, Any]: LLMChain 的完整输入。
        """
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        return {**kwargs, **new_inputs}

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。

        :meta private:
        """
        return list(set(self.llm_chain.input_keys) - {"agent_scratchpad"})

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        """验证提示是否匹配格式。

        参数:
            values: 要验证的值。

        返回:
            Dict: 验证后的值。

        异常:
            ValueError: 如果 `agent_scratchpad` 不在 prompt.input_variables 中
             并且提示不是 FewShotPromptTemplate 或 PromptTemplate。
        """
        prompt = self.llm_chain.prompt
        if "agent_scratchpad" not in prompt.input_variables:
            logger.warning(
                "`agent_scratchpad` 应该是 prompt.input_variables 中的一个变量。"
                " 未找到，因此在末尾添加。",
            )
            prompt.input_variables.append("agent_scratchpad")
            if isinstance(prompt, PromptTemplate):
                prompt.template += "\n{agent_scratchpad}"
            elif isinstance(prompt, FewShotPromptTemplate):
                prompt.suffix += "\n{agent_scratchpad}"
            else:
                msg = f"获取了意外的提示类型 {type(prompt)}"
                raise ValueError(msg)
        return self

    @property
    @abstractmethod
    def observation_prefix(self) -> str:
        """用于附加观察结果的前缀。"""

    @property
    @abstractmethod
    def llm_prefix(self) -> str:
        """用于附加 LLM 调用的前缀。"""

    @classmethod
    @abstractmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """为此类创建提示。

        参数:
            tools: 要使用的工具。

        返回:
            BasePromptTemplate: 提示模板。
        """

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """验证是否传入了适当的工具。

        参数:
            tools: 要使用的工具。
        """

    @classmethod
    @abstractmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        """获取此类的默认输出解析器。"""

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        output_parser: Optional[AgentOutputParser] = None,
        **kwargs: Any,
    ) -> Agent:
        """从 LLM 和工具构建代理。

        参数:
            llm: 要使用的语言模型。
            tools: 要使用的工具。
            callback_manager: 要使用的回调管理器。
            output_parser: 要使用的输出解析器。
            kwargs: 额外参数。

        返回:
            Agent: 代理对象。
        """
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=cls.create_prompt(tools),
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    def return_stopped_response(
        self,
        early_stopping_method: str,
        intermediate_steps: list[tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> AgentFinish:
        """当代理因达到最大迭代次数而停止时返回响应。

        参数:
            early_stopping_method: 用于提前停止的方法。
            intermediate_steps: LLM迄今为止采取的步骤，
                以及观察结果。
            **kwargs: 用户输入。

        返回:
            AgentFinish: 代理完成对象。

        异常:
            ValueError: 如果 `early_stopping_method` 不在 ['force', 'generate'] 中。
        """
        if early_stopping_method == "force":
            # `force` 只返回一个常量字符串
            return AgentFinish(
                {"output": "Agent stopped due to iteration limit or time limit."},
                "",
            )
        if early_stopping_method == "generate":
            # Generate 执行最后一次前向传播
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += (
                    f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                )
            # 在之前的步骤基础上，我们现在告诉 LLM 做出最终预测
            thoughts += (
                "\n\nI now need to return a final answer based on the previous steps:"
            )
            new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
            full_inputs = {**kwargs, **new_inputs}
            full_output = self.llm_chain.predict(**full_inputs)
            # 我们尝试提取最终答案
            parsed_output = self.output_parser.parse(full_output)
            if isinstance(parsed_output, AgentFinish):
                # 如果可以提取，我们就发送正确的内容
                return parsed_output
            # 如果可以提取，但工具不是最终工具，
            # 我们只返回完整输出
            return AgentFinish({"output": full_output}, full_output)
        msg = (
            "early_stopping_method 应该是 `force` 或 `generate` 之一，"
            f"但得到了 {early_stopping_method}"
        )
        raise ValueError(msg)

    def tool_run_logging_kwargs(self) -> builtins.dict:
        """返回工具运行的日志记录参数。"""
        return {
            "llm_prefix": self.llm_prefix,
            "observation_prefix": self.observation_prefix,
        }


class ExceptionTool(BaseTool):
    """仅返回查询的工具。"""

    name: str = "_Exception"
    """工具的名称。"""
    description: str = "Exception tool"
    """工具的描述。"""

    @override
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return query

    @override
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return query


NextStepOutput = list[Union[AgentFinish, AgentAction, AgentStep]]
RunnableAgentType = Union[RunnableAgent, RunnableMultiActionAgent]


class AgentExecutor(Chain):
    """使用工具的代理。"""

    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent, Runnable]
    """用于创建计划并在执行循环的每一步确定要采取的行动的代理。"""
    tools: Sequence[BaseTool]
    """代理可以调用的有效工具。"""
    return_intermediate_steps: bool = False
    """除了最终输出之外，是否在最后返回代理的中间步骤轨迹。"""
    max_iterations: Optional[int] = 15
    """在结束执行循环之前要采取的最大步数。

    设置为 'None' 可能导致无限循环。"""
    max_execution_time: Optional[float] = None
    """在执行循环中花费的最大时钟时间。"""
    early_stopping_method: str = "force"
    """如果代理从未返回 `AgentFinish`，则用于提前停止的方法。可以是 'force' 或 'generate'。

    `"force"` 返回一个字符串，说明它因为达到时间或迭代限制而停止。

    `"generate"` 最后一次调用代理的 LLM 链以根据先前的步骤生成最终答案。
    """
    handle_parsing_errors: Union[bool, str, Callable[[OutputParserException], str]] = (
        False
    )
    """如何处理代理输出解析器引发的错误。
    默认为 `False`，这将引发错误。
    如果为 `True`，错误将作为观察结果发送回 LLM。
    如果为字符串，该字符串本身将作为观察结果发送给 LLM。
    如果为可调用函数，该函数将以异常作为参数被调用，并且该函数的结果将作为观察结果传递给代理。
    """
    trim_intermediate_steps: Union[
        int,
        Callable[[list[tuple[AgentAction, str]]], list[tuple[AgentAction, str]]],
    ] = -1
    """在返回中间步骤之前如何修剪它们。
    默认为 -1，表示不修剪。
    """

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent, Runnable],
        tools: Sequence[BaseTool],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """从代理和工具创建。

        参数:
            agent: 要使用的代理。
            tools: 要使用的工具。
            callbacks: 要使用的回调。
            kwargs: 额外参数。

        返回:
            AgentExecutor: 代理执行器对象。
        """
        return cls(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            **kwargs,
        )

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        """验证工具是否与代理兼容。

        参数:
            values: 要验证的值。

        返回:
            Dict: 验证后的值。

        异常:
            ValueError: 如果允许的工具与提供的工具不同。
        """
        agent = self.agent
        tools = self.tools
        allowed_tools = agent.get_allowed_tools()  # type: ignore[union-attr]
        if allowed_tools is not None and set(allowed_tools) != {
            tool.name for tool in tools
        }:
            msg = (
                f"允许的工具 ({allowed_tools}) 与提供的工具 "
                f"({[tool.name for tool in tools]}) 不同"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_runnable_agent(cls, values: dict) -> Any:
        """如果传入，则将 runnable 转换为代理。

        参数:
            values: 要验证的值。

        返回:
            Dict: 验证后的值。
        """
        agent = values.get("agent")
        if agent and isinstance(agent, Runnable):
            try:
                output_type = agent.OutputType
            except TypeError:
                multi_action = False
            except Exception:
                logger.exception("从代理获取 OutputType 时发生意外错误")
                multi_action = False
            else:
                multi_action = output_type == Union[list[AgentAction], AgentFinish]

            stream_runnable = values.pop("stream_runnable", True)
            if multi_action:
                values["agent"] = RunnableMultiActionAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
            else:
                values["agent"] = RunnableAgent(
                    runnable=agent,
                    stream_runnable=stream_runnable,
                )
        return values

    @property
    def _action_agent(self) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
        """类型转换 self.agent。

        如果 `agent` 属性是 Runnable，它将在 validate_runnable_agent 根验证器中转换为 RunnableAgentType 之一。

        为了支持使用 Runnable 进行实例化，我们在这里显式地转换类型以反映根验证器中所做的更改。
        """
        if isinstance(self.agent, Runnable):
            return cast("RunnableAgentType", self.agent)
        return self.agent

    @override
    def save(self, file_path: Union[Path, str]) -> None:
        """引发错误 - 代理执行器不支持保存。

        参数:
            file_path: 保存路径。

        异常:
            ValueError: 代理执行器不支持保存。
        """
        msg = (
            "代理执行器不支持保存。"
            "如果您尝试保存代理，请使用 "
            "`.save_agent(...)`"
        )
        raise ValueError(msg)

    def save_agent(self, file_path: Union[Path, str]) -> None:
        """保存底层代理。

        参数:
            file_path: 保存路径。
        """
        return self._action_agent.save(file_path)

    def iter(
        self,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        include_run_info: bool = False,
        async_: bool = False,  # noqa: ARG002 arg kept for backwards compat, but ignored
    ) -> AgentExecutorIterator:
        """启用对达到最终输出所采取步骤的迭代。

        参数:
            inputs: 代理的输入。
            callbacks: 要运行的回调。
            include_run_info: 是否包含运行信息。
            async_: 是否异步运行。(已忽略)

        返回:
            AgentExecutorIterator: 代理执行器迭代器对象。
        """
        return AgentExecutorIterator(
            self,
            inputs,
            callbacks,
            tags=self.tags,
            include_run_info=include_run_info,
        )

    @property
    def input_keys(self) -> list[str]:
        """返回输入键。

        :meta private:
        """
        return self._action_agent.input_keys

    @property
    def output_keys(self) -> list[str]:
        """返回单个输出键。

        :meta private:
        """
        if self.return_intermediate_steps:
            return [*self._action_agent.return_values, "intermediate_steps"]
        return self._action_agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """按名称查找工具。

        参数:
            name: 工具名称。

        返回:
            BaseTool: 工具对象。
        """
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        return self.max_execution_time is None or time_elapsed < self.max_execution_time

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output,
                color="green",
                verbose=self.verbose,
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _consume_next_step(
        self,
        values: NextStepOutput,
    ) -> Union[AgentFinish, list[tuple[AgentAction, str]]]:
        if isinstance(values[-1], AgentFinish):
            if len(values) != 1:
                msg = "预期只有一个 AgentFinish 输出，但收到了多个值。"
                raise ValueError(msg)
            return values[-1]
        return [(a.action, a.observation) for a in values if isinstance(a, AgentStep)]

    def _take_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, list[tuple[AgentAction, str]]]:
        return self._consume_next_step(
            list(
                self._iter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                ),
            ),
        )

    def _iter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """在思考-动作-观察循环中执行单步操作。

        重写此方法可以控制代理如何做出选择并采取行动。
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # 调用 LLM 查看下一步做什么。
            output = self._action_agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "发生输出解析错误。"
                    "为了将此错误传回给代理并让其重试，"
                    "请将 `handle_parsing_errors=True` 传递给 AgentExecutor。"
                    f"错误信息如下: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "无效或不完整的响应"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "获取了意外的 `handle_parsing_errors` 类型"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # 如果选择的工具是结束工具，则我们结束并返回。
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action
        for agent_action in actions:
            yield self._perform_agent_action(
                name_to_tool_map,
                color_mapping,
                agent_action,
                run_manager,
            )

    def _perform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> AgentStep:
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        # 否则我们查找工具
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # 然后我们用工具输入调用工具以获取观察结果
            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    async def _atake_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, list[tuple[AgentAction, str]]]:
        return self._consume_next_step(
            [
                a
                async for a in self._aiter_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                )
            ],
        )

    async def _aiter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """在思考-动作-观察循环中执行单步操作。

        重写此方法可以控制代理如何做出选择并采取行动。
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # 调用 LLM 查看下一步做什么。
            output = await self._action_agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                msg = (
                    "发生输出解析错误。"
                    "为了将此错误传回给代理并让其重试，"
                    "请将 `handle_parsing_errors=True` 传递给 AgentExecutor。"
                    f"错误信息如下: {e!s}"
                )
                raise ValueError(msg) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "无效或不完整的响应"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                msg = "获取了意外的 `handle_parsing_errors` 类型"  # type: ignore[unreachable]
                raise ValueError(msg) from e  # noqa: TRY004
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        # 如果选择的工具是结束工具，则我们结束并返回。
        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        for agent_action in actions:
            yield agent_action

        # 使用 asyncio.gather 并发运行多个 tool.arun() 调用
        result = await asyncio.gather(
            *[
                self._aperform_agent_action(
                    name_to_tool_map,
                    color_mapping,
                    agent_action,
                    run_manager,
                )
                for agent_action in actions
            ],
        )

        # TODO: 这可以在每个结果可用时产生它
        for chunk in result:
            yield chunk

    async def _aperform_agent_action(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action,
                verbose=self.verbose,
                color="green",
            )
        # 否则我们查找工具
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # 然后我们用工具输入调用工具以获取观察结果
            observation = await tool.arun(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        else:
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = await InvalidTool().arun(
                {
                    "requested_tool_name": agent_action.tool,
                    "available_tool_names": list(name_to_tool_map.keys()),
                },
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
        return AgentStep(action=agent_action, observation=observation)

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        """运行文本并获取代理响应。"""
        # 构建工具名称到工具的映射以便于查找
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # 我们构建从每个工具到颜色的映射，用于日志记录。
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green", "red"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
        # 让我们开始跟踪迭代次数和经过的时间
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # 我们现在进入代理循环（直到它返回某些内容）。
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output,
                    intermediate_steps,
                    run_manager=run_manager,
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # 查看工具是否应直接返回
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self._action_agent.return_stopped_response(
            self.early_stopping_method,
            intermediate_steps,
            **inputs,
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> dict[str, str]:
        """异步运行文本并获取代理响应。"""
        # 构建工具名称到工具的映射以便于查找
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # 我们构建从每个工具到颜色的映射，用于日志记录。
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools],
            excluded_colors=["green"],
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []
        # 让我们开始跟踪迭代次数和经过的时间
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # 我们现在进入代理循环（直到它返回某些内容）。
        try:
            async with asyncio_timeout(self.max_execution_time):
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )

                    intermediate_steps.extend(next_step_output)
                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                        # 查看工具是否应直接返回
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return,
                                intermediate_steps,
                                run_manager=run_manager,
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                output = self._action_agent.return_stopped_response(
                    self.early_stopping_method,
                    intermediate_steps,
                    **inputs,
                )
                return await self._areturn(
                    output,
                    intermediate_steps,
                    run_manager=run_manager,
                )
        except (TimeoutError, asyncio.TimeoutError):
            # 异步超时中断时提前停止
            output = self._action_agent.return_stopped_response(
                self.early_stopping_method,
                intermediate_steps,
                **inputs,
            )
            return await self._areturn(
                output,
                intermediate_steps,
                run_manager=run_manager,
            )

    def _get_tool_return(
        self,
        next_step_output: tuple[AgentAction, str],
    ) -> Optional[AgentFinish]:
        """检查工具是否为返回工具。"""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        return_value_key = "output"
        if len(self._action_agent.return_values) > 0:
            return_value_key = self._action_agent.return_values[0]
        # 无效工具将不在映射中，因此我们返回 False。
        if (
            agent_action.tool in name_to_tool_map
            and name_to_tool_map[agent_action.tool].return_direct
        ):
            return AgentFinish(
                {return_value_key: observation},
                "",
            )
        return None

    def _prepare_intermediate_steps(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> list[tuple[AgentAction, str]]:
        """
        准备中间步骤列表，根据配置对步骤进行修剪。
        
        这个方法的主要目的是控制传递给LLM的上下文长度，防止因为中间步骤过多
        导致上下文过长，从而影响LLM的性能和成本。
        
        参数:
            intermediate_steps: 代理执行过程中的所有中间步骤列表
                每个元素是 (AgentAction, observation) 的元组
            
        返回:
            修剪后的中间步骤列表，用于传递给LLM作为上下文
        """
        # 情况1: 如果 trim_intermediate_steps 是正整数
        # 只保留最近的 N 个步骤，丢弃较早的步骤
        if (
            isinstance(self.trim_intermediate_steps, int)
            and self.trim_intermediate_steps > 0
        ):
            # 使用切片操作获取列表的最后 N 个元素
            # 例如：intermediate_steps[-5:] 获取最后5个步骤
            return intermediate_steps[-self.trim_intermediate_steps :]
        
        # 情况2: 如果 trim_intermediate_steps 是可调用函数
        # 使用自定义函数来修剪步骤（更灵活的修剪策略）
        if callable(self.trim_intermediate_steps):
            # 调用用户提供的修剪函数，让用户自定义如何选择要保留的步骤
            return self.trim_intermediate_steps(intermediate_steps)
        
        # 情况3: 默认情况（trim_intermediate_steps = -1 或其他值）
        # 不进行修剪，返回原始的完整步骤列表
        return intermediate_steps

    @override
    def stream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[AddableDict]:
        """启用对达到最终输出所采取步骤的流式处理。

        参数:
            input: 代理的输入。
            config: 要使用的配置。
            kwargs: 额外参数。

        产生:
            AddableDict: 可添加的字典。
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        yield from iterator

    @override
    async def astream(
        self,
        input: Union[dict[str, Any], Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AddableDict]:
        """异步启用对达到最终输出所采取步骤的流式处理。

        参数:
            input: 代理的输入。
            config: 要使用的配置。
            kwargs: 额外参数。

        产生:
            AddableDict: 可添加的字典。
        """
        config = ensure_config(config)
        iterator = AgentExecutorIterator(
            self,
            input,
            config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id"),
            yield_actions=True,
            **kwargs,
        )
        async for step in iterator:
            yield step
