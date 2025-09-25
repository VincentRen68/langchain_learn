"""用于表示代理动作、观察结果和返回值的模式定义。

**注意** 这些模式定义是为了向后兼容而提供的。

    新的代理应该使用 langgraph 库构建
    (https://github.com/langchain-ai/langgraph))，它提供了更简单
    和更灵活的方式来定义代理。

    请参阅迁移指南了解如何将现有代理迁移到现代 langgraph 代理：
    https://python.langchain.com/docs/how_to/migrate_agent/

代理使用语言模型来选择要执行的动作序列。

基本代理的工作方式如下：

1. 给定提示，代理使用 LLM 请求要执行的动作
   (例如，运行工具)。
2. 代理执行动作（例如，运行工具），并接收观察结果。
3. 代理将观察结果返回给 LLM，然后可以用来生成
   下一个动作。
4. 当代理达到停止条件时，它返回最终返回值。

代理本身的模式定义在 langchain.agents.agent 中。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, Union

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)


class AgentAction(Serializable):
    """表示代理执行动作的请求。

    动作包括要执行的工具名称和传递给工具的输入。
    日志用于传递关于动作的额外信息。
    """

    tool: str
    """要执行的工具名称。"""
    tool_input: Union[str, dict]
    """传递给工具的输入。"""
    log: str
    """关于动作的额外日志信息。
    这个日志可以用于几种方式。首先，它可以用于审计
    LLM 预测导致此 (tool, tool_input) 的确切内容。
    其次，它可以在未来的迭代中用于显示 LLM 的先前
    思考。这在 (tool, tool_input) 不包含
    LLM 预测的完整信息时很有用（例如，工具/工具输入之前的任何 `thought`）。"""
    type: Literal["AgentAction"] = "AgentAction"

    # 重写 init 以支持按位置实例化，用于向后兼容。
    def __init__(
        self, tool: str, tool_input: Union[str, dict], log: str, **kwargs: Any
    ):
        """创建 AgentAction。

        参数:
            tool: 要执行的工具名称。
            tool_input: 传递给工具的输入。
            log: 关于动作的额外日志信息。
        """
        super().__init__(tool=tool, tool_input=tool_input, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """AgentAction 是可序列化的。

        返回:
            True
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 langchain 对象的命名空间。

        返回:
            ``["langchain", "schema", "agent"]``
        """
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """返回与此动作对应的消息。"""
        return _convert_agent_action_to_messages(self)


class AgentActionMessageLog(AgentAction):
    """表示代理要执行的动作。

    这与 AgentAction 类似，但包含由聊天消息组成的消息日志。
    这在处理 ChatModels 时很有用，用于从代理的角度重建对话历史。
    """

    message_log: Sequence[BaseMessage]
    """类似于 log，这可以用于传递关于 LLM 在解析出 (tool, tool_input) 之前
    预测的确切消息的额外信息。这在 (tool, tool_input) 不能用于完全重建
    LLM 预测，而您需要该 LLM 预测（用于未来的代理迭代）时再次有用。
    与 `log` 相比，这在底层 LLM 是 ChatModel（因此返回消息而不是字符串）时很有用。"""
    # 忽略类型，因为我们正在重写 AgentAction 的类型。
    # 在这种情况下这是正确的做法。
    # 类型字面量用于序列化目的。
    type: Literal["AgentActionMessageLog"] = "AgentActionMessageLog"  # type: ignore[assignment]


class AgentStep(Serializable):
    """运行 AgentAction 的结果。"""

    action: AgentAction
    """已执行的 AgentAction。"""
    observation: Any
    """AgentAction 的结果。"""

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """与此观察结果对应的消息。"""
        return _convert_agent_observation_to_messages(self.action, self.observation)


class AgentFinish(Serializable):
    """ActionAgent 的最终返回值。

    当代理达到停止条件时，它们返回 AgentFinish。
    """

    return_values: dict
    """返回值的字典。"""
    log: str
    """关于返回值的额外日志信息。
    这用于传递完整的 LLM 预测，而不仅仅是解析出的
    返回值。例如，如果完整的 LLM 预测是
    `Final Answer: 2`，您可能只想返回 `2` 作为返回值，但传递
    完整字符串作为 `log`（用于调试或可观察性目的）。
    """
    type: Literal["AgentFinish"] = "AgentFinish"

    def __init__(self, return_values: dict, log: str, **kwargs: Any):
        """重写 init 以支持按位置实例化，用于向后兼容。"""
        super().__init__(return_values=return_values, log=log, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 True，因为此类是可序列化的。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 langchain 对象的命名空间。

        返回:
            ``["langchain", "schema", "agent"]``
        """
        return ["langchain", "schema", "agent"]

    @property
    def messages(self) -> Sequence[BaseMessage]:
        """与此观察结果对应的消息。"""
        return [AIMessage(content=self.log)]


def _convert_agent_action_to_messages(
    agent_action: AgentAction,
) -> Sequence[BaseMessage]:
    """将代理动作转换为消息。

    此代码用于从代理动作重建原始 AI 消息。

    参数:
        agent_action: 要转换的代理动作。

    返回:
        对应于原始工具调用的 AIMessage。
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return agent_action.message_log
    return [AIMessage(content=agent_action.log)]


def _convert_agent_observation_to_messages(
    agent_action: AgentAction, observation: Any
) -> Sequence[BaseMessage]:
    """将代理动作转换为消息。

    此代码用于从代理动作重建原始 AI 消息。

    参数:
        agent_action: 要转换的代理动作。
        observation: 要转换为消息的观察结果。

    返回:
        对应于原始工具调用的 AIMessage。
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return [_create_function_message(agent_action, observation)]
    content = observation
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    return [HumanMessage(content=content)]


def _create_function_message(
    agent_action: AgentAction, observation: Any
) -> FunctionMessage:
    """将代理动作和观察结果转换为函数消息。

    参数:
        agent_action: 来自代理的工具调用请求。
        observation: 工具调用的结果。

    返回:
        对应于原始工具调用的 FunctionMessage。
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )
