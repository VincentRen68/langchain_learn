"""与工具相关的消息。"""

import json
from typing import Any, Literal, Optional, Union
from uuid import UUID

from pydantic import Field, model_validator
from typing_extensions import NotRequired, TypedDict, override

from langchain_core.messages.base import BaseMessage, BaseMessageChunk, merge_content
from langchain_core.utils._merge import merge_dicts, merge_obj


class ToolOutputMixin:
    """可供工具直接返回的对象的混入类 (Mixin)。

    如果一个自定义的 BaseTool 被 ToolCall 调用，且其自定义代码的输出
    不是 ToolOutputMixin 的实例，那么该输出将被自动强制转换成字符串
    并包装在一个 ToolMessage 中。
    """


class ToolMessage(BaseMessage, ToolOutputMixin):
    """用于将执行工具后的结果传递回模型的消息。

    ToolMessage 包含工具调用的结果。通常，结果被编码在 `content` 字段中。

    示例：一个 ToolMessage，表示 ID 为 call_Jja7J89XsjrOLA5r!MEOW!SL 的工具调用返回了结果 42

        .. code-block:: python

            from langchain_core.messages import ToolMessage

            ToolMessage(content="42", tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL")


    示例：一个 ToolMessage，其中只有部分工具输出被发送给模型，
        而完整的输出则通过 artifact 参数传递。

        .. versionadded:: 0.2.17

        .. code-block:: python

            from langchain_core.messages import ToolMessage

            tool_output = {
                "stdout": "从图中我们可以看到 x 和 y 之间的相关性是...",
                "stderr": None,
                "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},
            }

            ToolMessage(
                content=tool_output["stdout"],
                artifact=tool_output,
                tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
            )

    `tool_call_id` 字段用于将工具调用请求与工具调用响应关联起来。
    这在聊天模型能够并行请求多个工具调用的情况下非常有用。

    """

    tool_call_id: str
    """此消息所响应的工具调用的 ID。"""

    type: Literal["tool"] = "tool"
    """消息的类型（用于反序列化）。默认为 "tool"。"""

    artifact: Any = None
    """工具执行的产物（artifact），这部分内容不会发送给模型。

    仅当它与消息内容不同时才应指定，例如，只有工具完整输出的子集
    作为消息内容传递，但代码的其他部分需要完整的输出时。

    .. versionadded:: 0.2.17
    """

    status: Literal["success", "error"] = "success"
    """工具调用的状态。

    .. versionadded:: 0.2.24
    """

    additional_kwargs: dict = Field(default_factory=dict, repr=False)
    """当前继承自 BaseMessage，但未使用。"""
    response_metadata: dict = Field(default_factory=dict, repr=False)
    """当前继承自 BaseMessage，但未使用。"""

    @model_validator(mode="before")
    @classmethod
    def coerce_args(cls, values: dict) -> dict:
        """将模型参数强制转换为正确的类型。

        参数:
            values: 模型参数。
        """
        content = values["content"]
        if isinstance(content, tuple):
            content = list(content)

        if not isinstance(content, (str, list)):
            try:
                values["content"] = str(content)
            except ValueError as e:
                msg = (
                    "ToolMessage content should be a string or a list of string/dicts. "
                    f"Received:\n\n{content=}\n\n which could not be coerced into a "
                    "string."
                )
                raise ValueError(msg) from e
        elif isinstance(content, list):
            values["content"] = []
            for i, x in enumerate(content):
                if not isinstance(x, (str, dict)):
                    try:
                        values["content"].append(str(x))
                    except ValueError as e:
                        msg = (
                            "ToolMessage content should be a string or a list of "
                            "string/dicts. Received a list but "
                            f"element ToolMessage.content[{i}] is not a dict and could "
                            f"not be coerced to a string.:\n\n{x}"
                        )
                        raise ValueError(msg) from e
                else:
                    values["content"].append(x)

        tool_call_id = values["tool_call_id"]
        if isinstance(tool_call_id, (UUID, int, float)):
            values["tool_call_id"] = str(tool_call_id)
        return values

    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        """创建一个 ToolMessage。

        参数:
            content: 消息的字符串内容。
            **kwargs: 其他字段。
        """
        super().__init__(content=content, **kwargs)


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """工具消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                msg = "Cannot concatenate ToolMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                artifact=merge_obj(self.artifact, other.artifact),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
                status=_merge_status(self.status, other.status),
            )

        return super().__add__(other)


class ToolCall(TypedDict):
    """表示一个调用工具的请求。

    示例:

        .. code-block:: python

            {"name": "foo", "args": {"a": 1}, "id": "123"}

        这表示一个请求，要求调用名为 "foo" 的工具，参数为 {"a": 1}，
        并且其标识符为 "123"。

    """

    name: str
    """要调用的工具的名称。"""
    args: dict[str, Any]
    """传递给工具调用的参数。"""
    id: Optional[str]
    """与工具调用关联的标识符。

    当有多个并发的工具调用时，需要一个标识符来将工具调用请求
    与工具调用结果关联起来。
    """
    type: NotRequired[Literal["tool_call"]]


def tool_call(
    *,
    name: str,
    args: dict[str, Any],
    id: Optional[str],
) -> ToolCall:
    """创建一个工具调用。

    参数:
        name: 要调用的工具的名称。
        args: 传递给工具调用的参数。
        id: 与工具调用关联的标识符。

    返回:
        创建的工具调用对象。
    """
    return ToolCall(name=name, args=args, id=id, type="tool_call")


class ToolCallChunk(TypedDict):
    """一个工具调用的块（例如，作为流的一部分）。

    当合并 ToolCallChunk 时（例如，通过 AIMessageChunk.__add__），
    所有字符串属性都会被拼接。只有当块的 `index` 值相等且不为 None 时，
    它们才会被合并。

    示例:

    .. code-block:: python

        left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
        right_chunks = [ToolCallChunk(name=None, args="1}", index=0)]

        (
            AIMessageChunk(content="", tool_call_chunks=left_chunks)
            + AIMessageChunk(content="", tool_call_chunks=right_chunks)
        ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]

    """

    name: Optional[str]
    """要调用的工具的名称。"""
    args: Optional[str]
    """传递给工具调用的参数。"""
    id: Optional[str]
    """与工具调用关联的标识符。"""
    index: Optional[int]
    """工具调用在序列中的索引。"""
    type: NotRequired[Literal["tool_call_chunk"]]


def tool_call_chunk(
    *,
    name: Optional[str] = None,
    args: Optional[str] = None,
    id: Optional[str] = None,
    index: Optional[int] = None,
) -> ToolCallChunk:
    """创建一个工具调用块。

    参数:
        name: 要调用的工具的名称。
        args: 传递给工具调用的参数。
        id: 与工具调用关联的标识符。
        index: 工具调用在序列中的索引。

    返回:
        创建的工具调用块对象。
    """
    return ToolCallChunk(
        name=name, args=args, id=id, index=index, type="tool_call_chunk"
    )


class InvalidToolCall(TypedDict):
    """为 LLM 可能犯的错误提供兼容空间。

    这里我们添加一个 `error` 键，用于暴露在生成过程中发生的错误
    （例如，无效的 JSON 参数）。
    """

    name: Optional[str]
    """要调用的工具的名称。"""
    args: Optional[str]
    """传递给工具调用的参数。"""
    id: Optional[str]
    """与工具调用关联的标识符。"""
    error: Optional[str]
    """与工具调用关联的错误信息。"""
    type: NotRequired[Literal["invalid_tool_call"]]


def invalid_tool_call(
    *,
    name: Optional[str] = None,
    args: Optional[str] = None,
    id: Optional[str] = None,
    error: Optional[str] = None,
) -> InvalidToolCall:
    """创建一个无效的工具调用。

    参数:
        name: 要调用的工具的名称。
        args: 传递给工具调用的参数。
        id: 与工具调用关联的标识符。
        error: 与工具调用关联的错误信息。

    返回:
        创建的无效工具调用对象。
    """
    return InvalidToolCall(
        name=name, args=args, id=id, error=error, type="invalid_tool_call"
    )


def default_tool_parser(
    raw_tool_calls: list[dict],
) -> tuple[list[ToolCall], list[InvalidToolCall]]:
    """尽最大努力解析工具调用。

    参数:
        raw_tool_calls: 待解析的原始工具调用字典列表。

    返回:
        一个包含有效工具调用和无效工具调用的元组。
    """
    tool_calls = []
    invalid_tool_calls = []
    for raw_tool_call in raw_tool_calls:
        if "function" not in raw_tool_call:
            continue
        function_name = raw_tool_call["function"]["name"]
        try:
            function_args = json.loads(raw_tool_call["function"]["arguments"])
            parsed = tool_call(
                name=function_name or "",
                args=function_args or {},
                id=raw_tool_call.get("id"),
            )
            tool_calls.append(parsed)
        except json.JSONDecodeError:
            invalid_tool_calls.append(
                invalid_tool_call(
                    name=function_name,
                    args=raw_tool_call["function"]["arguments"],
                    id=raw_tool_call.get("id"),
                    error=None,
                )
            )
    return tool_calls, invalid_tool_calls


def default_tool_chunk_parser(raw_tool_calls: list[dict]) -> list[ToolCallChunk]:
    """尽最大努力解析工具调用块。

    参数:
        raw_tool_calls: 待解析的原始工具调用字典列表。

    返回:
        解析后的 ToolCallChunk 对象列表。
    """
    tool_call_chunks = []
    for tool_call in raw_tool_calls:
        if "function" not in tool_call:
            function_args = None
            function_name = None
        else:
            function_args = tool_call["function"]["arguments"]
            function_name = tool_call["function"]["name"]
        parsed = tool_call_chunk(
            name=function_name,
            args=function_args,
            id=tool_call.get("id"),
            index=tool_call.get("index"),
        )
        tool_call_chunks.append(parsed)
    return tool_call_chunks


def _merge_status(
    left: Literal["success", "error"], right: Literal["success", "error"]
) -> Literal["success", "error"]:
    return "error" if "error" in {left, right} else "success"
