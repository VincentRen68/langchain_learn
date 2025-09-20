"""此模块包含用于处理消息的实用函数。

使用这些函数可以实现以下功能：

* 将消息转换为字符串（序列化）
* 将字典转换为 Message 对象（反序列化）
* 根据名称、类型或 ID 等从消息列表中筛选消息。
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import math
from collections.abc import Iterable, Sequence
from functools import partial
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)

from pydantic import Discriminator, Field, Tag

from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.messages import convert_to_openai_data_block, is_data_content_block
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolCall, ToolMessage, ToolMessageChunk

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables.base import Runnable

try:
    from langchain_text_splitters import TextSplitter

    _HAS_LANGCHAIN_TEXT_SPLITTERS = True
except ImportError:
    _HAS_LANGCHAIN_TEXT_SPLITTERS = False

logger = logging.getLogger(__name__)


def _get_type(v: Any) -> str:
    """获取与对象关联的、用于序列化的类型。"""
    if isinstance(v, dict) and "type" in v:
        return v["type"]
    if hasattr(v, "type"):
        return v.type
    msg = (
        f"Expected either a dictionary with a 'type' key or an object "
        f"with a 'type' attribute. Instead got type {type(v)}."
    )
    raise TypeError(msg)


AnyMessage = Annotated[
    Union[
        Annotated[AIMessage, Tag(tag="ai")],
        Annotated[HumanMessage, Tag(tag="human")],
        Annotated[ChatMessage, Tag(tag="chat")],
        Annotated[SystemMessage, Tag(tag="system")],
        Annotated[FunctionMessage, Tag(tag="function")],
        Annotated[ToolMessage, Tag(tag="tool")],
        Annotated[AIMessageChunk, Tag(tag="AIMessageChunk")],
        Annotated[HumanMessageChunk, Tag(tag="HumanMessageChunk")],
        Annotated[ChatMessageChunk, Tag(tag="ChatMessageChunk")],
        Annotated[SystemMessageChunk, Tag(tag="SystemMessageChunk")],
        Annotated[FunctionMessageChunk, Tag(tag="FunctionMessageChunk")],
        Annotated[ToolMessageChunk, Tag(tag="ToolMessageChunk")],
    ],
    Field(discriminator=Discriminator(_get_type)),
]


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    r"""将一个消息序列转换为字符串，并将它们拼接成一个单一的字符串。

    参数:
        messages: 需要被转换为字符串的消息列表。
        human_prefix: 添加在 HumanMessage 内容前的缀。默认为 "Human"。
        ai_prefix: 添加在 AIMessage 内容前的缀。默认为 "AI"。

    返回:
        一个由所有输入消息拼接而成的单一字符串。

    异常:
        ValueError: 如果遇到不支持的消息类型。

    示例:
        .. code-block:: python

            from langchain_core import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="你好，最近怎么样？"),
                AIMessage(content="我很好，你呢？"),
            ]
            get_buffer_string(messages)
            # -> "Human: 你好，最近怎么样？\nAI: 我很好，你呢？"

    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ToolMessage):
            role = "Tool"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            msg = f"Got unsupported message type: {m}"
            raise ValueError(msg)  # noqa: TRY004
        message = f"{role}: {m.text()}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)


def _message_from_dict(message: dict) -> BaseMessage:
    type_ = message["type"]
    if type_ == "human":
        return HumanMessage(**message["data"])
    if type_ == "ai":
        return AIMessage(**message["data"])
    if type_ == "system":
        return SystemMessage(**message["data"])
    if type_ == "chat":
        return ChatMessage(**message["data"])
    if type_ == "function":
        return FunctionMessage(**message["data"])
    if type_ == "tool":
        return ToolMessage(**message["data"])
    if type_ == "remove":
        return RemoveMessage(**message["data"])
    if type_ == "AIMessageChunk":
        return AIMessageChunk(**message["data"])
    if type_ == "HumanMessageChunk":
        return HumanMessageChunk(**message["data"])
    if type_ == "FunctionMessageChunk":
        return FunctionMessageChunk(**message["data"])
    if type_ == "ToolMessageChunk":
        return ToolMessageChunk(**message["data"])
    if type_ == "SystemMessageChunk":
        return SystemMessageChunk(**message["data"])
    if type_ == "ChatMessageChunk":
        return ChatMessageChunk(**message["data"])
    msg = f"Got unexpected message type: {type_}"
    raise ValueError(msg)


def messages_from_dict(messages: Sequence[dict]) -> list[BaseMessage]:
    """将一个消息字典序列转换为 Message 对象列表。

    参数:
        messages: 需要转换的消息字典序列。

    返回:
        Message 对象（BaseMessage）的列表。
    """
    return [_message_from_dict(m) for m in messages]


def message_chunk_to_message(chunk: BaseMessage) -> BaseMessage:
    """将一个消息块（message chunk）转换为一个完整的消息（message）。

    参数:
        chunk: 需要转换的消息块。

    返回:
        完整的消息对象。
    """
    if not isinstance(chunk, BaseMessageChunk):
        return chunk
    # chunk 类总是将其对应的非 chunk 类作为其第一个父类
    ignore_keys = ["type"]
    if isinstance(chunk, AIMessageChunk):
        ignore_keys.append("tool_call_chunks")
    return chunk.__class__.__mro__[1](
        **{k: v for k, v in chunk.__dict__.items() if k not in ignore_keys}
    )


MessageLikeRepresentation = Union[
    BaseMessage, list[str], tuple[str, str], str, dict[str, Any]
]


def _create_message_from_message_type(
    message_type: str,
    content: str,
    name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    id: Optional[str] = None,
    **additional_kwargs: Any,
) -> BaseMessage:
    """根据消息类型和内容字符串创建一个消息对象。

    参数:
        message_type: (str) 消息的类型 (例如, "human", "ai" 等)。
        content: (str) 内容字符串。
        name: (str) 消息的名称。默认为 None。
        tool_call_id: (str) 工具调用的 ID。默认为 None。
        tool_calls: (list[dict[str, Any]]) 工具调用列表。默认为 None。
        id: (str) 消息的 ID。默认为 None。
        additional_kwargs: (dict[str, Any]) 其他关键字参数。

    返回:
        一个相应类型的消息对象。

    异常:
        ValueError: 如果消息类型不是 "human", "user", "ai",
            "assistant", "function", "tool", "system", 或 "developer" 之一。
    """
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if tool_call_id is not None:
        kwargs["tool_call_id"] = tool_call_id
    if additional_kwargs:
        if response_metadata := additional_kwargs.pop("response_metadata", None):
            kwargs["response_metadata"] = response_metadata
        kwargs["additional_kwargs"] = additional_kwargs
        additional_kwargs.update(additional_kwargs.pop("additional_kwargs", {}))
    if id is not None:
        kwargs["id"] = id
    if tool_calls is not None:
        kwargs["tool_calls"] = []
        for tool_call in tool_calls:
            # Convert OpenAI-format tool call to LangChain format.
            if "function" in tool_call:
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args, strict=False)
                kwargs["tool_calls"].append(
                    {
                        "name": tool_call["function"]["name"],
                        "args": args,
                        "id": tool_call["id"],
                        "type": "tool_call",
                    }
                )
            else:
                kwargs["tool_calls"].append(tool_call)
    if message_type in {"human", "user"}:
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message: BaseMessage = HumanMessage(content=content, **kwargs)
    elif message_type in {"ai", "assistant"}:
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message = AIMessage(content=content, **kwargs)
    elif message_type in {"system", "developer"}:
        if message_type == "developer":
            kwargs["additional_kwargs"] = kwargs.get("additional_kwargs") or {}
            kwargs["additional_kwargs"]["__openai_role__"] = "developer"
        message = SystemMessage(content=content, **kwargs)
    elif message_type == "function":
        message = FunctionMessage(content=content, **kwargs)
    elif message_type == "tool":
        artifact = kwargs.get("additional_kwargs", {}).pop("artifact", None)
        status = kwargs.get("additional_kwargs", {}).pop("status", None)
        if status is not None:
            kwargs["status"] = status
        message = ToolMessage(content=content, artifact=artifact, **kwargs)
    elif message_type == "remove":
        message = RemoveMessage(**kwargs)
    else:
        msg = (
            f"Unexpected message type: '{message_type}'. Use one of 'human',"
            f" 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'."
        )
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise ValueError(msg)
    return message


def _convert_to_message(message: MessageLikeRepresentation) -> BaseMessage:
    """从多种消息格式中实例化一个消息对象。

    消息格式可以是以下之一:

    - BaseMessagePromptTemplate
    - BaseMessage
    - (角色字符串, 模板) 的二元组; 例如, ("human", "{user_input}")
    - dict: 包含 role 和 content 键的消息字典
    - string: ("human", 模板) 的简写; 例如, "{user_input}"

    参数:
        message: 一种受支持格式的消息表示。

    返回:
        一个消息或消息模板的实例。

    异常:
        NotImplementedError: 如果消息类型不受支持。
        ValueError: 如果消息字典不包含必需的键。
    """
    if isinstance(message, BaseMessage):
        message_ = message
    elif isinstance(message, str):
        message_ = _create_message_from_message_type("human", message)
    elif isinstance(message, Sequence) and len(message) == 2:
        # mypy 没有意识到，鉴于之前的分支，这不可能是一个字符串
        message_type_str, template = message  # type: ignore[misc]
        message_ = _create_message_from_message_type(message_type_str, template)
    elif isinstance(message, dict):
        msg_kwargs = message.copy()
        try:
            try:
                msg_type = msg_kwargs.pop("role")
            except KeyError:
                msg_type = msg_kwargs.pop("type")
            # 不允许 None 消息内容
            msg_content = msg_kwargs.pop("content") or ""
        except KeyError as e:
            msg = f"Message dict must contain 'role' and 'content' keys, got {message}"
            msg = create_message(
                message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE
            )
            raise ValueError(msg) from e
        message_ = _create_message_from_message_type(
            msg_type, msg_content, **msg_kwargs
        )
    else:
        msg = f"Unsupported message type: {type(message)}"
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise NotImplementedError(msg)

    return message_


def convert_to_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
) -> list[BaseMessage]:
    """将一个消息序列转换为消息列表。

    参数:
        messages: 需要转换的消息序列。

    返回:
        消息对象（BaseMessage）的列表。
    """
    # 在此处导入以避免循环导入
    from langchain_core.prompt_values import PromptValue  # noqa: PLC0415

    if isinstance(messages, PromptValue):
        return messages.to_messages()
    return [_convert_to_message(m) for m in messages]


def _runnable_support(func: Callable) -> Callable:
    @overload
    def wrapped(
        messages: None = None, **kwargs: Any
    ) -> Runnable[Sequence[MessageLikeRepresentation], list[BaseMessage]]: ...

    @overload
    def wrapped(
        messages: Sequence[MessageLikeRepresentation], **kwargs: Any
    ) -> list[BaseMessage]: ...

    def wrapped(
        messages: Union[Sequence[MessageLikeRepresentation], None] = None,
        **kwargs: Any,
    ) -> Union[
        list[BaseMessage],
        Runnable[Sequence[MessageLikeRepresentation], list[BaseMessage]],
    ]:
        # 在本地导入以避免循环导入。
        from langchain_core.runnables.base import RunnableLambda  # noqa: PLC0415

        if messages is not None:
            return func(messages, **kwargs)
        return RunnableLambda(partial(func, **kwargs), name=func.__name__)

    wrapped.__doc__ = func.__doc__
    return wrapped


@_runnable_support
def filter_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    include_names: Optional[Sequence[str]] = None,
    exclude_names: Optional[Sequence[str]] = None,
    include_types: Optional[Sequence[Union[str, type[BaseMessage]]]] = None,
    exclude_types: Optional[Sequence[Union[str, type[BaseMessage]]]] = None,
    include_ids: Optional[Sequence[str]] = None,
    exclude_ids: Optional[Sequence[str]] = None,
    exclude_tool_calls: Optional[Sequence[str] | bool] = None,
) -> list[BaseMessage]:
    """根据名称、类型或 ID 筛选消息。

    参数:
        messages: 需要筛选的类消息（Message-like）对象序列。
        include_names: 需要包含的消息名称。默认为 None。
        exclude_names: 需要排除的消息名称。默认为 None。
        include_types: 需要包含的消息类型。可以指定为字符串名称 (例如
            "system", "human", "ai", ...) 或 BaseMessage 类 (例如
            SystemMessage, HumanMessage, AIMessage, ...)。默认为 None。
        exclude_types: 需要排除的消息类型。可以指定为字符串名称 (例如
            "system", "human", "ai", ...) 或 BaseMessage 类 (例如
            SystemMessage, HumanMessage, AIMessage, ...)。默认为 None。
        include_ids: 需要包含的消息 ID。默认为 None。
        exclude_ids: 需要排除的消息 ID。默认为 None。
        exclude_tool_calls: 需要排除的工具调用 ID。默认为 None。
            可以是以下之一:

            - ``True``: 每个带有工具调用的 ``AIMessage`` 和所有的 ``ToolMessage``
              都将被排除。
            - 一个需要排除的工具调用 ID 序列:

              - 具有相应工具调用 ID 的 ToolMessage 将被排除。
              - AIMessage 中的 ``tool_calls`` 将被更新，以排除匹配的工具调用。
                如果一个 AIMessage 的所有 tool_calls 都被过滤掉，
                则整个消息将被排除。

    返回:
        一个消息列表，该列表中的消息满足至少一个 incl_* 条件且不满足任何
        excl_* 条件。如果没有指定 incl_* 条件，则任何未被明确排除的
        消息都将被包含。

    异常:
        ValueError: 如果提供了两个不兼容的参数。

    示例:
        .. code-block:: python

            from langchain_core.messages import (
                filter_messages,
                AIMessage,
                HumanMessage,
                SystemMessage,
            )

            messages = [
                SystemMessage("你是个好助手。"),
                HumanMessage("你叫什么名字", id="foo", name="example_user"),
                AIMessage("steve-o", id="bar", name="example_assistant"),
                HumanMessage(
                    "你最喜欢的颜色是什么",
                    id="baz",
                ),
                AIMessage(
                    "硅蓝色",
                    id="blah",
                ),
            ]

            filter_messages(
                messages,
                incl_names=("example_user", "example_assistant"),
                incl_types=("system",),
                excl_ids=("bar",),
            )

        .. code-block:: python

            [
                SystemMessage("你是个好助手。"),
                HumanMessage("你叫什么名字", id="foo", name="example_user"),
            ]

    """
    messages = convert_to_messages(messages)
    filtered: list[BaseMessage] = []
    for msg in messages:
        if (
            (exclude_names and msg.name in exclude_names)
            or (exclude_types and _is_message_type(msg, exclude_types))
            or (exclude_ids and msg.id in exclude_ids)
        ):
            continue

        if exclude_tool_calls is True and (
            (isinstance(msg, AIMessage) and msg.tool_calls)
            or isinstance(msg, ToolMessage)
        ):
            continue

        if isinstance(exclude_tool_calls, (list, tuple, set)):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls = [
                    tool_call
                    for tool_call in msg.tool_calls
                    if tool_call["id"] not in exclude_tool_calls
                ]
                if not tool_calls:
                    continue

                content = msg.content
                # 处理 Anthropic 内容块
                if isinstance(msg.content, list):
                    content = [
                        content_block
                        for content_block in msg.content
                        if (
                            not isinstance(content_block, dict)
                            or content_block.get("type") != "tool_use"
                            or content_block.get("id") not in exclude_tool_calls
                        )
                    ]

                msg = msg.model_copy(  # noqa: PLW2901
                    update={"tool_calls": tool_calls, "content": content}
                )
            elif (
                isinstance(msg, ToolMessage) and msg.tool_call_id in exclude_tool_calls
            ):
                continue

        # 在没有给出包含标准时，默认为包含。
        if (
            not (include_types or include_ids or include_names)
            or (include_names and msg.name in include_names)
            or (include_types and _is_message_type(msg, include_types))
            or (include_ids and msg.id in include_ids)
        ):
            filtered.append(msg)

    return filtered


@_runnable_support
def merge_message_runs(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    chunk_separator: str = "\n",
) -> list[BaseMessage]:
    r"""Merge consecutive Messages of the same type.

    **NOTE**: ToolMessages are not merged, as each has a distinct tool call id that
    can't be merged.

    Args:
        messages: Sequence Message-like objects to merge.
        chunk_separator: Specify the string to be inserted between message chunks.
        Default is "\n".

    Returns:
        list of BaseMessages with consecutive runs of message types merged into single
        messages. By default, if two messages being merged both have string contents,
        the merged content is a concatenation of the two strings with a new-line
        separator.
        The separator inserted between message chunks can be controlled by specifying
        any string with ``chunk_separator``. If at least one of the messages has a list
        of content blocks, the merged content is a list of content blocks.

    Example:

        .. code-block:: python

            from langchain_core.messages import (
                merge_message_runs,
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolCall,
            )

            messages = [
                SystemMessage("you're a good assistant."),
                HumanMessage(
                    "what's your favorite color",
                    id="foo",
                ),
                HumanMessage(
                    "wait your favorite food",
                    id="bar",
                ),
                AIMessage(
                    "my favorite colo",
                    tool_calls=[
                        ToolCall(
                            name="blah_tool", args={"x": 2}, id="123", type="tool_call"
                        )
                    ],
                    id="baz",
                ),
                AIMessage(
                    [{"type": "text", "text": "my favorite dish is lasagna"}],
                    tool_calls=[
                        ToolCall(
                            name="blah_tool",
                            args={"x": -10},
                            id="456",
                            type="tool_call",
                        )
                    ],
                    id="blur",
                ),
            ]

            merge_message_runs(messages)

        .. code-block:: python

            [
                SystemMessage("you're a good assistant."),
                HumanMessage(
                    "what's your favorite color\\n"
                    "wait your favorite food", id="foo",
                ),
                AIMessage(
                    [
                        "my favorite colo",
                        {"type": "text", "text": "my favorite dish is lasagna"}
                    ],
                    tool_calls=[
                        ToolCall({
                            "name": "blah_tool",
                            "args": {"x": 2},
                            "id": "123",
                            "type": "tool_call"
                        }),
                        ToolCall({
                            "name": "blah_tool",
                            "args": {"x": -10},
                            "id": "456",
                            "type": "tool_call"
                        })
                    ]
                    id="baz"
                ),
            ]

    """
    if not messages:
        return []
    messages = convert_to_messages(messages)
    merged: list[BaseMessage] = []
    for msg in messages:
        last = merged.pop() if merged else None
        if not last:
            merged.append(msg)
        elif isinstance(msg, ToolMessage) or not isinstance(msg, last.__class__):
            merged.extend([last, msg])
        else:
            last_chunk = _msg_to_chunk(last)
            curr_chunk = _msg_to_chunk(msg)
            if curr_chunk.response_metadata:
                curr_chunk.response_metadata.clear()
            if (
                isinstance(last_chunk.content, str)
                and isinstance(curr_chunk.content, str)
                and last_chunk.content
                and curr_chunk.content
            ):
                last_chunk.content += chunk_separator
            merged.append(_chunk_to_msg(last_chunk + curr_chunk))
    return merged


# TODO: 更新代码，使验证错误（例如，token_counter 的错误）在初始化时而不是在运行时引发。
@_runnable_support
def trim_messages(
    messages: Union[Iterable[MessageLikeRepresentation], PromptValue],
    *,
    max_tokens: int,
    token_counter: Union[
        Callable[[list[BaseMessage]], int],
        Callable[[BaseMessage], int],
        BaseLanguageModel,
    ],
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    start_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    include_system: bool = False,
    text_splitter: Optional[Union[Callable[[str], list[str]], TextSplitter]] = None,
) -> list[BaseMessage]:
    r"""将消息列表裁剪到指定的 token 数量以下。

    `trim_messages` 可用于将聊天记录的大小减少到指定的 token 数量或消息数量。

    无论哪种情况，如果将裁剪后的聊天记录直接传回给聊天模型，
    生成的聊天记录通常应满足以下属性：

    1. 生成的聊天记录应是有效的。大多数聊天模型期望聊天记录以
       (1) 一个 ``HumanMessage`` 或 (2) 一个 ``SystemMessage`` 后跟一个 ``HumanMessage`` 开始。
       为实现此目的，请设置 ``start_on="human"``。
       此外，通常 ``ToolMessage`` 只能出现在包含工具调用的 ``AIMessage`` 之后。
       更多关于消息的信息，请参阅以下链接：
       https://python.langchain.com/docs/concepts/#messages
    2. 它应包含最近的消息并丢弃聊天记录中较旧的消息。
       为实现此目的，请设置 ``strategy="last"``。
    3. 通常，新的聊天记录应包含原始聊天记录中的 ``SystemMessage``（如果存在），
       因为 ``SystemMessage`` 包含了对聊天模型的特殊指令。如果存在，
       ``SystemMessage`` 几乎总是历史记录中的第一条消息。为实现此目的，请设置
       ``include_system=True``。

    .. note::
        以下示例展示了如何配置 ``trim_messages`` 以实现与上述属性一致的行为。

    参数:
        messages: 需要裁剪的类消息（Message-like）对象序列。
        max_tokens: 裁剪后消息的最大 token 数量。
        token_counter: 用于计算 BaseMessage 或 BaseMessage 列表中的 token 的函数或 llm。
            如果传入一个 BaseLanguageModel，则将使用
            BaseLanguageModel.get_num_tokens_from_messages()。
            设置为 `len` 可计算聊天记录中的 **消息** 数量。

            .. note::
                使用 `count_tokens_approximately` 可获得快速、近似的 token 计数。
                建议在需要高性能的热路径上使用 `trim_messages` 时采用此方法，
                此时精确的 token 计数不是必需的。

        strategy: 裁剪策略。

            - "first": 保留消息的前 <= n_count 个 token。
            - "last": 保留消息的后 <= n_count 个 token。

            默认为 ``'last'``。
        allow_partial: 是否允许在只有部分消息可以被包含时拆分消息。
            如果 ``strategy="last"``，则包含消息的最后一部分内容。
            如果 ``strategy="first"``，则包含消息的第一部分内容。
            默认为 False。
        end_on: 结束的消息类型。如果指定，则该类型最后一次出现之后的所有消息都将被忽略。
            如果 ``strategy=="last"``，则在尝试获取最后 ``max_tokens`` 之前执行此操作。
            如果 ``strategy=="first"``，则在获取第一个 ``max_tokens`` 之后执行此操作。
            可以指定为字符串名称 (例如 "system", "human", "ai", ...) 或 BaseMessage 类
            (例如 SystemMessage, HumanMessage, AIMessage, ...)。可以是一种类型或一个类型列表。
            默认为 None。
        start_on: 开始的消息类型。仅当 ``strategy="last"`` 时才应指定。
            如果指定，则该类型第一次出现之前的所有消息都将被忽略。
            此操作在我们将初始消息裁剪到最后 ``max_tokens`` 之后执行。
            如果 ``include_system=True``，则不适用于索引为 0 的 SystemMessage。
            可以指定为字符串名称 (例如 "system", "human", "ai", ...) 或 BaseMessage 类
            (例如 SystemMessage, HumanMessage, AIMessage, ...)。可以是一种类型或一个类型列表。
            默认为 None。
        include_system: 是否保留索引为 0 的 SystemMessage（如果存在）。
            仅当 ``strategy="last"`` 时才应指定。
            默认为 False。
        text_splitter: 用于拆分消息字符串内容的函数或 ``langchain_text_splitters.TextSplitter``。
            仅在 ``allow_partial=True`` 时使用。如果 ``strategy="last"``，
            则将包含部分消息的最后被拆分的 token。如果 ``strategy=="first"``，
            则将包含部分消息的最前被拆分的 token。Token 拆分器假定保留了分隔符，
            以便可以直接拼接拆分后的内容以重新创建原始文本。默认为按换行符拆分。

    返回:
        裁剪后的 BaseMessage 列表。

    异常:
        ValueError: 如果指定了两个不兼容的参数或无法识别的 ``strategy``。

    示例:
        根据 token 数量裁剪聊天记录，保留 SystemMessage（如果存在），
        并确保聊天记录以 HumanMessage（或 SystemMessage 后跟 HumanMessage）开始。

        .. code-block:: python

            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                BaseMessage,
                SystemMessage,
                trim_messages,
            )

            messages = [
                SystemMessage(
                    "你是个好助手，你总是用笑话来回应。"
                ),
                HumanMessage("我想知道为什么它叫 langchain"),
                AIMessage(
                    '嗯，我猜他们觉得“WordRope”和“SentenceString”听起来没那么酷吧！'
                ),
                HumanMessage("那哈里森到底在追谁"),
                AIMessage(
                    "嗯，让我想想。\n\n哦，他可能是在追办公室里最后一杯咖啡！"
                ),
                HumanMessage("不会说话的鹦鹉叫什么"),
            ]


            trim_messages(
                messages,
                max_tokens=45,
                strategy="last",
                token_counter=ChatOpenAI(model="gpt-4o"),
                # 大多数聊天模型期望聊天记录以以下任一方式开始:
                # (1) 一个 HumanMessage 或
                # (2) 一个 SystemMessage 后跟一个 HumanMessage
                start_on="human",
                # 通常，我们希望保留 SystemMessage
                # 如果它存在于原始历史记录中。
                # SystemMessage 包含了对模型的特殊指令。
                include_system=True,
                allow_partial=False,
            )

        .. code-block:: python

            [
                SystemMessage(
                    content="你是个好助手，你总是用笑话来回应。"
                ),
                HumanMessage(content="不会说话的鹦鹉叫什么"),
            ]

        根据消息数量裁剪聊天记录，保留 SystemMessage（如果存在），
        并确保聊天记录以 HumanMessage（或 SystemMessage 后跟 HumanMessage）开始。

            trim_messages(
                messages,
                # 当 `len` 作为 token 计数器函数传入时，
                # max_tokens 将计算聊天记录中的消息数量。
                max_tokens=4,
                strategy="last",
                # 传入 `len` 作为 token 计数器函数将
                # 计算聊天记录中的消息数量。
                token_counter=len,
                # 大多数聊天模型期望聊天记录以以下任一方式开始:
                # (1) 一个 HumanMessage 或
                # (2) 一个 SystemMessage 后跟一个 HumanMessage
                start_on="human",
                # 通常，我们希望保留 SystemMessage
                # 如果它存在于原始历史记录中。
                # SystemMessage 包含了对模型的特殊指令。
                include_system=True,
                allow_partial=False,
            )

        .. code-block:: python

            [
                SystemMessage(
                    content="你是个好助手，你总是用笑话来回应。"
                ),
                HumanMessage(content="那哈里森到底在追谁"),
                AIMessage(
                    content="嗯，让我想想。\n\n哦，他可能是在追办公室里最后一杯咖啡！"
                ),
                HumanMessage(content="不会说话的鹦鹉叫什么"),
            ]


        使用一个计算每条消息中 token 数量的自定义 token 计数器函数来裁剪聊天记录。

        .. code-block:: python

            messages = [
                SystemMessage("这是一个 4 token 的文本。完整的消息是 10 个 token。"),
                HumanMessage(
                    "这是一个 4 token 的文本。完整的消息是 10 个 token。", id="first"
                ),
                AIMessage(
                    [
                        {"type": "text", "text": "这是第一个 4 token 的块。"},
                        {"type": "text", "text": "这是第二个 4 token 的块。"},
                    ],
                    id="second",
                ),
                HumanMessage(
                    "这是一个 4 token 的文本。完整的消息是 10 个 token。", id="third"
                ),
                AIMessage(
                    "这是一个 4 token 的文本。完整的消息是 10 个 token。",
                    id="fourth",
                ),
            ]


            def dummy_token_counter(messages: list[BaseMessage]) -> int:
                # 将每条消息视为在消息开头和结尾各添加了 3 个默认 token。
                # 3 + 4 + 3 = 每条消息 10 个 token。

                default_content_len = 4
                default_msg_prefix_len = 3
                default_msg_suffix_len = 3

                count = 0
                for msg in messages:
                    if isinstance(msg.content, str):
                        count += (
                            default_msg_prefix_len
                            + default_content_len
                            + default_msg_suffix_len
                        )
                    if isinstance(msg.content, list):
                        count += (
                            default_msg_prefix_len
                            + len(msg.content) * default_content_len
                            + default_msg_suffix_len
                        )
                return count

        前 30 个 token，允许部分消息:
            .. code-block:: python

                trim_messages(
                    messages,
                    max_tokens=30,
                    token_counter=dummy_token_counter,
                    strategy="first",
                    allow_partial=True,
                )

            .. code-block:: python

                [
                    SystemMessage(
                        "这是一个 4 token 的文本。完整的消息是 10 个 token。"
                    ),
                    HumanMessage(
                        "这是一个 4 token 的文本。完整的消息是 10 个 token。",
                        id="first",
                    ),
                    AIMessage(
                        [{"type": "text", "text": "这是第一个 4 token 的块。"}],
                        id="second",
                    ),
                ]

    """
    # 验证参数
    if start_on and strategy == "first":
        msg = "start_on parameter is only valid with strategy='last'"
        raise ValueError(msg)
    if include_system and strategy == "first":
        msg = "include_system parameter is only valid with strategy='last'"
        raise ValueError(msg)

    messages = convert_to_messages(messages)
    if hasattr(token_counter, "get_num_tokens_from_messages"):
        list_token_counter = token_counter.get_num_tokens_from_messages
    elif callable(token_counter):
        if (
            next(iter(inspect.signature(token_counter).parameters.values())).annotation
            is BaseMessage
        ):

            def list_token_counter(messages: Sequence[BaseMessage]) -> int:
                return sum(token_counter(msg) for msg in messages)  # type: ignore[arg-type, misc]

        else:
            list_token_counter = token_counter
    else:
        msg = (
            f"'token_counter' expected to be a model that implements "
            f"'get_num_tokens_from_messages()' or a function. Received object of type "
            f"{type(token_counter)}."
        )
        raise ValueError(msg)

    if _HAS_LANGCHAIN_TEXT_SPLITTERS and isinstance(text_splitter, TextSplitter):
        text_splitter_fn = text_splitter.split_text
    elif text_splitter:
        text_splitter_fn = cast("Callable", text_splitter)
    else:
        text_splitter_fn = _default_text_splitter

    if strategy == "first":
        return _first_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            text_splitter=text_splitter_fn,
            partial_strategy="first" if allow_partial else None,
            end_on=end_on,
        )
    if strategy == "last":
        return _last_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            include_system=include_system,
            start_on=start_on,
            end_on=end_on,
            text_splitter=text_splitter_fn,
        )
    msg = f"Unrecognized {strategy=}. Supported strategies are 'last' and 'first'."
    raise ValueError(msg)


def convert_to_openai_messages(
    messages: Union[MessageLikeRepresentation, Sequence[MessageLikeRepresentation]],
    *,
    text_format: Literal["string", "block"] = "string",
) -> Union[dict, list[dict]]:
    """将 LangChain 消息转换为 OpenAI 消息字典。

    参数:
        messages: 类消息对象或其可迭代对象，其内容格式可以为
            OpenAI、Anthropic、Bedrock Converse 或 VertexAI 格式。
        text_format: 如何格式化字符串或文本块内容:

            - ``'string'``:
              如果消息内容是字符串，则保持为字符串。如果消息
              包含的内容块全部为 'text' 类型，则用换行符将它们连接成
              一个单一的字符串。如果消息包含内容块且至少有一个不是
              'text' 类型，则所有块都保持为字典。
            - ``'block'``:
              如果消息内容是字符串，则将其转换为一个包含单个 'text'
              类型内容块的列表。如果消息包含内容块，则保持不变。

    异常:
        ValueError: 如果指定了无法识别的 ``text_format``，或者消息
            内容块缺少预期的键。

    返回:
        返回类型取决于输入类型:

        - dict:
          如果传入单个类消息对象，则返回单个 OpenAI 消息字典。
        - list[dict]:
          如果传入一个类消息对象序列，则返回一个 OpenAI 消息字典列表。

    示例:

        .. code-block:: python

            from langchain_core.messages import (
                convert_to_openai_messages,
                AIMessage,
                SystemMessage,
                ToolMessage,
            )

            messages = [
                SystemMessage([{"type": "text", "text": "foo"}]),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这里面有什么"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,'/9j/4AAQSk'"},
                        },
                    ],
                },
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "analyze",
                            "args": {"baz": "buz"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage("foobar", tool_call_id="1", name="bar"),
                {"role": "assistant", "content": "那很好"},
            ]
            oai_messages = convert_to_openai_messages(messages)
            # -> [
            #   {'role': 'system', 'content': 'foo'},
            #   {'role': 'user', 'content': [{'type': 'text', 'text': '这里面有什么'}, {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}]},
            #   {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': '1','function': {'name': 'analyze', 'arguments': '{"baz": "buz"}'}}], 'content': ''},
            #   {'role': 'tool', 'name': 'bar', 'content': 'foobar'},
            #   {'role': 'assistant', 'content': '那很好'}
            # ]

    .. versionadded:: 0.3.11

    """  # noqa: E501
    if text_format not in {"string", "block"}:
        err = f"Unrecognized {text_format=}, expected one of 'string' or 'block'."
        raise ValueError(err)

    oai_messages: list = []

    if is_single := isinstance(messages, (BaseMessage, dict, str)):
        messages = [messages]

    messages = convert_to_messages(messages)

    for i, message in enumerate(messages):
        oai_msg: dict = {"role": _get_message_openai_role(message)}
        tool_messages: list = []
        content: Union[str, list[dict]]

        if message.name:
            oai_msg["name"] = message.name
        if isinstance(message, AIMessage) and message.tool_calls:
            oai_msg["tool_calls"] = _convert_to_openai_tool_calls(message.tool_calls)
        if message.additional_kwargs.get("refusal"):
            oai_msg["refusal"] = message.additional_kwargs["refusal"]
        if isinstance(message, ToolMessage):
            oai_msg["tool_call_id"] = message.tool_call_id

        if not message.content:
            content = "" if text_format == "string" else []
        elif isinstance(message.content, str):
            if text_format == "string":
                content = message.content
            else:
                content = [{"type": "text", "text": message.content}]
        elif text_format == "string" and all(
            isinstance(block, str) or block.get("type") == "text"
            for block in message.content
        ):
            content = "\n".join(
                block if isinstance(block, str) else block["text"]
                for block in message.content
            )
        else:
            content = []
            for j, block in enumerate(message.content):
                # OpenAI format
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                elif block.get("type") == "text":
                    if missing := [k for k in ("text",) if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'text' "
                            f"but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    content.append({"type": block["type"], "text": block["text"]})
                elif block.get("type") == "image_url":
                    if missing := [k for k in ("image_url",) if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'image_url' "
                            f"but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": block["image_url"],
                        }
                    )
                # Standard multi-modal content block
                elif is_data_content_block(block):
                    formatted_block = convert_to_openai_data_block(block)
                    if (
                        formatted_block.get("type") == "file"
                        and "file" in formatted_block
                        and "filename" not in formatted_block["file"]
                    ):
                        logger.info("Generating a fallback filename.")
                        formatted_block["file"]["filename"] = "LC_AUTOGENERATED"
                    content.append(formatted_block)
                # Anthropic and Bedrock converse format
                elif (block.get("type") == "image") or "image" in block:
                    # Anthropic
                    if source := block.get("source"):
                        if missing := [
                            k for k in ("media_type", "type", "data") if k not in source
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image' "
                                f"but 'source' is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{source['media_type']};"
                                        f"{source['type']},{source['data']}"
                                    )
                                },
                            }
                        )
                    # Bedrock converse
                    elif image := block.get("image"):
                        if missing := [
                            k for k in ("source", "format") if k not in image
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has key 'image', "
                                f"but 'image' is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        b64_image = _bytes_to_b64_str(image["source"]["bytes"])
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:image/{image['format']};base64,{b64_image}"
                                    )
                                },
                            }
                        )
                    else:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'image' "
                            f"but does not have a 'source' or 'image' key. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(err)
                # OpenAI file format
                elif (
                    block.get("type") == "file"
                    and isinstance(block.get("file"), dict)
                    and isinstance(block.get("file", {}).get("file_data"), str)
                ):
                    if block.get("file", {}).get("filename") is None:
                        logger.info("Generating a fallback filename.")
                        block["file"]["filename"] = "LC_AUTOGENERATED"
                    content.append(block)
                # OpenAI audio format
                elif (
                    block.get("type") == "input_audio"
                    and isinstance(block.get("input_audio"), dict)
                    and isinstance(block.get("input_audio", {}).get("data"), str)
                    and isinstance(block.get("input_audio", {}).get("format"), str)
                ):
                    content.append(block)
                elif block.get("type") == "tool_use":
                    if missing := [
                        k for k in ("id", "name", "input") if k not in block
                    ]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'tool_use', but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    if not any(
                        tool_call["id"] == block["id"]
                        for tool_call in cast("AIMessage", message).tool_calls
                    ):
                        oai_msg["tool_calls"] = oai_msg.get("tool_calls", [])
                        oai_msg["tool_calls"].append(
                            {
                                "type": "function",
                                "id": block["id"],
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(
                                        block["input"], ensure_ascii=False
                                    ),
                                },
                            }
                        )
                elif block.get("type") == "tool_result":
                    if missing := [
                        k for k in ("content", "tool_use_id") if k not in block
                    ]:
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'tool_result', but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    tool_message = ToolMessage(
                        block["content"],
                        tool_call_id=block["tool_use_id"],
                        status="error" if block.get("is_error") else "success",
                    )
                    # Recurse to make sure tool message contents are OpenAI format.
                    tool_messages.extend(
                        convert_to_openai_messages(
                            [tool_message], text_format=text_format
                        )
                    )
                elif (block.get("type") == "json") or "json" in block:
                    if "json" not in block:
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'json' "
                            f"but does not have a 'json' key. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    content.append(
                        {
                            "type": "text",
                            "text": json.dumps(block["json"]),
                        }
                    )
                elif (block.get("type") == "guard_content") or "guard_content" in block:
                    if (
                        "guard_content" not in block
                        or "text" not in block["guard_content"]
                    ):
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'guard_content' but does not have a "
                            f"messages[{i}].content[{j}]['guard_content']['text'] "
                            f"key. Full content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    text = block["guard_content"]["text"]
                    if isinstance(text, dict):
                        text = text["text"]
                    content.append({"type": "text", "text": text})
                # VertexAI format
                elif block.get("type") == "media":
                    if missing := [k for k in ("mime_type", "data") if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'media' but does not have key(s) {missing}. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    if "image" not in block["mime_type"]:
                        err = (
                            f"OpenAI messages can only support text and image data."
                            f" Received content block with media of type:"
                            f" {block['mime_type']}"
                        )
                        raise ValueError(err)
                    b64_image = _bytes_to_b64_str(block["data"])
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:{block['mime_type']};base64,{b64_image}")
                            },
                        }
                    )
                elif block.get("type") == "thinking":
                    content.append(block)
                else:
                    err = (
                        f"Unrecognized content block at "
                        f"messages[{i}].content[{j}] does not match OpenAI, "
                        f"Anthropic, Bedrock Converse, or VertexAI format. Full "
                        f"content block:\n\n{block}"
                    )
                    raise ValueError(err)
            if text_format == "string" and not any(
                block["type"] != "text" for block in content
            ):
                content = "\n".join(block["text"] for block in content)
        oai_msg["content"] = content
        if message.content and not oai_msg["content"] and tool_messages:
            oai_messages.extend(tool_messages)
        else:
            oai_messages.extend([oai_msg, *tool_messages])

    if is_single:
        return oai_messages[0]
    return oai_messages


def _first_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    partial_strategy: Optional[Literal["first", "last"]] = None,
    end_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if not messages:
        return messages

    # Check if all messages already fit within token limit
    if token_counter(messages) <= max_tokens:
        # When all messages fit, only apply end_on filtering if needed
        if end_on:
            for _ in range(len(messages)):
                if not _is_message_type(messages[-1], end_on):
                    messages.pop()
                else:
                    break
        return messages

    # Use binary search to find the maximum number of messages within token limit
    left, right = 0, len(messages)
    max_iterations = len(messages).bit_length()
    for _ in range(max_iterations):
        if left >= right:
            break
        mid = (left + right + 1) // 2
        if token_counter(messages[:mid]) <= max_tokens:
            left = mid
            idx = mid
        else:
            right = mid - 1

    # idx now contains the maximum number of complete messages we can include
    idx = left

    if partial_strategy and idx < len(messages):
        included_partial = False
        copied = False
        if isinstance(messages[idx].content, list):
            excluded = messages[idx].model_copy(deep=True)
            copied = True
            num_block = len(excluded.content)
            if partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
            for _ in range(1, num_block):
                excluded.content = excluded.content[:-1]
                if token_counter([*messages[:idx], excluded]) <= max_tokens:
                    messages = [*messages[:idx], excluded]
                    idx += 1
                    included_partial = True
                    break
            if included_partial and partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
        if not included_partial:
            if not copied:
                excluded = messages[idx].model_copy(deep=True)
                copied = True

            # Extract text content efficiently
            text = None
            if isinstance(excluded.content, str):
                text = excluded.content
            elif isinstance(excluded.content, list) and excluded.content:
                for block in excluded.content:
                    if isinstance(block, str):
                        text = block
                        break
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        break

            if text:
                if not copied:
                    excluded = excluded.model_copy(deep=True)

                split_texts = text_splitter(text)
                base_message_count = token_counter(messages[:idx])
                if partial_strategy == "last":
                    split_texts = list(reversed(split_texts))

                # Binary search for the maximum number of splits we can include
                left, right = 0, len(split_texts)
                max_iterations = len(split_texts).bit_length()
                for _ in range(max_iterations):
                    if left >= right:
                        break
                    mid = (left + right + 1) // 2
                    excluded.content = "".join(split_texts[:mid])
                    if base_message_count + token_counter([excluded]) <= max_tokens:
                        left = mid
                    else:
                        right = mid - 1

                if left > 0:
                    content_splits = split_texts[:left]
                    if partial_strategy == "last":
                        content_splits = list(reversed(content_splits))
                    excluded.content = "".join(content_splits)
                    messages = [*messages[:idx], excluded]
                    idx += 1

    if end_on:
        for _ in range(idx):
            if idx > 0 and not _is_message_type(messages[idx - 1], end_on):
                idx -= 1
            else:
                break

    return messages[:idx]


def _last_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    allow_partial: bool = False,
    include_system: bool = False,
    start_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
    end_on: Optional[
        Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]]
    ] = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if len(messages) == 0:
        return []

    # Filter out messages after end_on type
    if end_on:
        for _ in range(len(messages)):
            if not _is_message_type(messages[-1], end_on):
                messages.pop()
            else:
                break

    # Handle system message preservation
    system_message = None
    if include_system and len(messages) > 0 and isinstance(messages[0], SystemMessage):
        system_message = messages[0]
        messages = messages[1:]

    # Reverse messages to use _first_max_tokens with reversed logic
    reversed_messages = messages[::-1]

    # Calculate remaining tokens after accounting for system message if present
    remaining_tokens = max_tokens
    if system_message:
        system_tokens = token_counter([system_message])
        remaining_tokens = max(0, max_tokens - system_tokens)

    reversed_result = _first_max_tokens(
        reversed_messages,
        max_tokens=remaining_tokens,
        token_counter=token_counter,
        text_splitter=text_splitter,
        partial_strategy="last" if allow_partial else None,
        end_on=start_on,
    )

    # Re-reverse the messages and add back the system message if needed
    result = reversed_result[::-1]
    if system_message:
        result = [system_message, *result]

    return result


_MSG_CHUNK_MAP: dict[type[BaseMessage], type[BaseMessageChunk]] = {
    HumanMessage: HumanMessageChunk,
    AIMessage: AIMessageChunk,
    SystemMessage: SystemMessageChunk,
    ToolMessage: ToolMessageChunk,
    FunctionMessage: FunctionMessageChunk,
    ChatMessage: ChatMessageChunk,
}
_CHUNK_MSG_MAP = {v: k for k, v in _MSG_CHUNK_MAP.items()}


def _msg_to_chunk(message: BaseMessage) -> BaseMessageChunk:
    if message.__class__ in _MSG_CHUNK_MAP:
        return _MSG_CHUNK_MAP[message.__class__](**message.model_dump(exclude={"type"}))

    for msg_cls, chunk_cls in _MSG_CHUNK_MAP.items():
        if isinstance(message, msg_cls):
            return chunk_cls(**message.model_dump(exclude={"type"}))

    msg = (
        f"Unrecognized message class {message.__class__}. Supported classes are "
        f"{list(_MSG_CHUNK_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _chunk_to_msg(chunk: BaseMessageChunk) -> BaseMessage:
    if chunk.__class__ in _CHUNK_MSG_MAP:
        return _CHUNK_MSG_MAP[chunk.__class__](
            **chunk.model_dump(exclude={"type", "tool_call_chunks"})
        )
    for chunk_cls, msg_cls in _CHUNK_MSG_MAP.items():
        if isinstance(chunk, chunk_cls):
            return msg_cls(**chunk.model_dump(exclude={"type", "tool_call_chunks"}))

    msg = (
        f"Unrecognized message chunk class {chunk.__class__}. Supported classes are "
        f"{list(_CHUNK_MSG_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _default_text_splitter(text: str) -> list[str]:
    splits = text.split("\n")
    return [s + "\n" for s in splits[:-1]] + splits[-1:]


def _is_message_type(
    message: BaseMessage,
    type_: Union[str, type[BaseMessage], Sequence[Union[str, type[BaseMessage]]]],
) -> bool:
    types = [type_] if isinstance(type_, (str, type)) else type_
    types_str = [t for t in types if isinstance(t, str)]
    types_types = tuple(t for t in types if isinstance(t, type))

    return message.type in types_str or isinstance(message, types_types)


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


def _get_message_openai_role(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        return message.additional_kwargs.get("__openai_role__", "system")
    if isinstance(message, FunctionMessage):
        return "function"
    if isinstance(message, ChatMessage):
        return message.role
    msg = f"Unknown BaseMessage type {message.__class__}."
    raise ValueError(msg)


def _convert_to_openai_tool_calls(tool_calls: list[ToolCall]) -> list[dict]:
    return [
        {
            "type": "function",
            "id": tool_call["id"],
            "function": {
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
            },
        }
        for tool_call in tool_calls
    ]


def count_tokens_approximately(
    messages: Iterable[MessageLikeRepresentation],
    *,
    chars_per_token: float = 4.0,
    extra_tokens_per_message: float = 3.0,
    count_name: bool = True,
) -> int:
    """Approximate the total number of tokens in messages.

    The token count includes stringified message content, role, and (optionally) name.
    - For AI messages, the token count also includes stringified tool calls.
    - For tool messages, the token count also includes the tool call ID.

    Args:
        messages: List of messages to count tokens for.
        chars_per_token: Number of characters per token to use for the approximation.
            Default is 4 (one token corresponds to ~4 chars for common English text).
            You can also specify float values for more fine-grained control.
            `See more here. <https://platform.openai.com/tokenizer>`__
        extra_tokens_per_message: Number of extra tokens to add per message.
            Default is 3 (special tokens, including beginning/end of message).
            You can also specify float values for more fine-grained control.
            `See more here. <https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>`__
        count_name: Whether to include message names in the count.
            Enabled by default.

    Returns:
        Approximate number of tokens in the messages.

    .. note::
        This is a simple approximation that may not match the exact token count used by
        specific models. For accurate counts, use model-specific tokenizers.

    Warning:
        This function does not currently support counting image tokens.

    .. versionadded:: 0.3.46

    """
    token_count = 0.0
    for message in convert_to_messages(messages):
        message_chars = 0
        if isinstance(message.content, str):
            message_chars += len(message.content)

        # TODO: add support for approximate counting for image blocks
        else:
            content = repr(message.content)
            message_chars += len(content)

        if (
            isinstance(message, AIMessage)
            # exclude Anthropic format as tool calls are already included in the content
            and not isinstance(message.content, list)
            and message.tool_calls
        ):
            tool_calls_content = repr(message.tool_calls)
            message_chars += len(tool_calls_content)

        if isinstance(message, ToolMessage):
            message_chars += len(message.tool_call_id)

        role = _get_message_openai_role(message)
        message_chars += len(role)

        if message.name and count_name:
            message_chars += len(message.name)

        # NOTE: we're rounding up per message to ensure that
        # individual message token counts add up to the total count
        # for a list of messages
        token_count += math.ceil(message_chars / chars_per_token)

        # add extra tokens per message
        token_count += extra_tokens_per_message

    # round up once more time in case extra_tokens_per_message is a float
    return math.ceil(token_count)
