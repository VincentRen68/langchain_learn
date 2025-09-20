"""AI 消息。"""

import json
import logging
import operator
from typing import Any, Literal, Optional, Union, cast

from pydantic import model_validator
from typing_extensions import NotRequired, Self, TypedDict, override

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.messages.tool import (
    InvalidToolCall,
    ToolCall,
    ToolCallChunk,
    default_tool_chunk_parser,
    default_tool_parser,
)
from langchain_core.messages.tool import (
    invalid_tool_call as create_invalid_tool_call,
)
from langchain_core.messages.tool import (
    tool_call as create_tool_call,
)
from langchain_core.messages.tool import (
    tool_call_chunk as create_tool_call_chunk,
)
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.json import parse_partial_json
from langchain_core.utils.usage import _dict_int_op

logger = logging.getLogger(__name__)


_LC_ID_PREFIX = "run-"


class InputTokenDetails(TypedDict, total=False):
    """输入 token 数量的明细。

    各部分总和*不必*等于总输入 token 数。*不必*包含所有键。

    示例:

        .. code-block:: python

            {
                "audio": 10,
                "cache_creation": 200,
                "cache_read": 100,
            }

    .. versionadded:: 0.3.9

    也可能包含提供商特定的额外键。

    """

    audio: int
    """音频输入的 token。"""
    cache_creation: int
    """被缓存但缓存未命中的输入 token。

    由于缓存未命中，缓存是根据这些 token 创建的。
    """
    cache_read: int
    """被缓存且缓存命中的输入 token。

    由于缓存命中，这些 token 是从缓存中读取的。更准确地说，
    是给定这些 token 后的模型状态是从缓存中读取的。
    """


class OutputTokenDetails(TypedDict, total=False):
    """输出 token 数量的明细。

    各部分总和*不必*等于总输出 token 数。*不必*包含所有键。

    示例:

        .. code-block:: python

            {
                "audio": 10,
                "reasoning": 200,
            }

    .. versionadded:: 0.3.9

    """

    audio: int
    """音频输出的 token。"""
    reasoning: int
    """推理过程输出的 token。

    指模型在思维链（chain of thought）过程中生成的、但不会作为最终模型输出返回的 token
    （例如 OpenAI 的 o1 模型产生的 token）。
    """


class UsageMetadata(TypedDict):
    """消息的使用情况元数据，例如 token 数量。

    这是一种跨模型一致的 token 使用情况的标准表示。

    示例:

        .. code-block:: python

            {
                "input_tokens": 350,
                "output_tokens": 240,
                "total_tokens": 590,
                "input_token_details": {
                    "audio": 10,
                    "cache_creation": 200,
                    "cache_read": 100,
                },
                "output_token_details": {
                    "audio": 10,
                    "reasoning": 200,
                },
            }

    .. versionchanged:: 0.3.9

        添加了 ``input_token_details`` 和 ``output_token_details``。

    """

    input_tokens: int
    """输入（或提示）的 token 数量。是所有输入 token 类型的总和。"""
    output_tokens: int
    """输出（或补全）的 token 数量。是所有输出 token 类型的总和。"""
    total_tokens: int
    """总 token 数量。是 input_tokens + output_tokens 的总和。"""
    input_token_details: NotRequired[InputTokenDetails]
    """输入 token 数量的明细。

    各部分总和*不必*等于总输入 token 数。*不必*包含所有键。
    """
    output_token_details: NotRequired[OutputTokenDetails]
    """输出 token 数量的明细。

    各部分总和*不必*等于总输出 token 数。*不必*包含所有键。
    """


class AIMessage(BaseMessage):
    """来自 AI 的消息。

    AIMessage 是聊天模型对提示（prompt）作出响应后返回的消息。

    此消息代表模型的输出，它既包含模型返回的原始输出，
    也包含由 LangChain 框架添加的标准化字段（例如，工具调用、使用情况元数据）。
    """

    example: bool = False
    """用于表示此消息是一段示例对话的一部分。

    目前，大多数模型会忽略此字段。不建议使用。
    """

    tool_calls: list[ToolCall] = []
    """如果提供，则为与此消息关联的工具调用。"""
    invalid_tool_calls: list[InvalidToolCall] = []
    """如果提供，则为与此消息关联的、存在解析错误的工具调用。"""
    usage_metadata: Optional[UsageMetadata] = None
    """如果提供，则为消息的使用情况元数据，例如 token 数量。

    这是一种跨模型一致的 token 使用情况的标准表示。
    """

    type: Literal["ai"] = "ai"
    """消息的类型（用于反序列化）。默认为 "ai"。"""

    def __init__(
        self, content: Union[str, list[Union[str, dict]]], **kwargs: Any
    ) -> None:
        """将 content 作为位置参数传入。

        参数:
            content: 消息的内容。
            kwargs: 传递给父类的其他参数。
        """
        super().__init__(content=content, **kwargs)

    @property
    def lc_attributes(self) -> dict:
        """即使是从其他初始化参数派生的属性，也需要被序列化。"""
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    # TODO: remove this logic if possible, reducing breaking nature of changes
    @model_validator(mode="before")
    @classmethod
    def _backwards_compat_tool_calls(cls, values: dict) -> Any:
        check_additional_kwargs = not any(
            values.get(k)
            for k in ("tool_calls", "invalid_tool_calls", "tool_call_chunks")
        )
        if check_additional_kwargs and (
            raw_tool_calls := values.get("additional_kwargs", {}).get("tool_calls")
        ):
            try:
                if issubclass(cls, AIMessageChunk):
                    values["tool_call_chunks"] = default_tool_chunk_parser(
                        raw_tool_calls
                    )
                else:
                    parsed_tool_calls, parsed_invalid_tool_calls = default_tool_parser(
                        raw_tool_calls
                    )
                    values["tool_calls"] = parsed_tool_calls
                    values["invalid_tool_calls"] = parsed_invalid_tool_calls
            except Exception:
                logger.debug("Failed to parse tool calls", exc_info=True)

        # Ensure "type" is properly set on all tool call-like dicts.
        if tool_calls := values.get("tool_calls"):
            values["tool_calls"] = [
                create_tool_call(**{k: v for k, v in tc.items() if k != "type"})
                for tc in tool_calls
            ]
        if invalid_tool_calls := values.get("invalid_tool_calls"):
            values["invalid_tool_calls"] = [
                create_invalid_tool_call(**{k: v for k, v in tc.items() if k != "type"})
                for tc in invalid_tool_calls
            ]

        if tool_call_chunks := values.get("tool_call_chunks"):
            values["tool_call_chunks"] = [
                create_tool_call_chunk(**{k: v for k, v in tc.items() if k != "type"})
                for tc in tool_call_chunks
            ]

        return values

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """返回消息的美化表示。

        参数:
            html: 是否返回 HTML 格式的字符串。
                  默认为 False。

        返回:
            消息的美化表示。
        """
        base = super().pretty_repr(html=html)
        lines = []

        def _format_tool_args(tc: Union[ToolCall, InvalidToolCall]) -> list[str]:
            lines = [
                f"  {tc.get('name', 'Tool')} ({tc.get('id')})",
                f" Call ID: {tc.get('id')}",
            ]
            if tc.get("error"):
                lines.append(f"  Error: {tc.get('error')}")
            lines.append("  Args:")
            args = tc.get("args")
            if isinstance(args, str):
                lines.append(f"    {args}")
            elif isinstance(args, dict):
                for arg, value in args.items():
                    lines.append(f"    {arg}: {value}")
            return lines

        if self.tool_calls:
            lines.append("Tool Calls:")
            for tc in self.tool_calls:
                lines.extend(_format_tool_args(tc))
        if self.invalid_tool_calls:
            lines.append("Invalid Tool Calls:")
            for itc in self.invalid_tool_calls:
                lines.extend(_format_tool_args(itc))
        return (base.strip() + "\n" + "\n".join(lines)).strip()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """来自 AI 的消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment]
    """消息的类型（用于反序列化）。
    默认为 "AIMessageChunk"。"""

    tool_call_chunks: list[ToolCallChunk] = []
    """如果提供，则为与此消息关联的工具调用块。"""

    @property
    def lc_attributes(self) -> dict:
        """即使是从其他初始化参数派生的属性，也需要被序列化。"""
        return {
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
        }

    @model_validator(mode="after")
    def init_tool_calls(self) -> Self:
        """从工具调用块（tool call chunks）初始化工具调用（tool calls）。

        返回:
            此 ``AIMessageChunk`` 对象。
        """
        if not self.tool_call_chunks:
            if self.tool_calls:
                self.tool_call_chunks = [
                    create_tool_call_chunk(
                        name=tc["name"],
                        args=json.dumps(tc["args"]),
                        id=tc["id"],
                        index=None,
                    )
                    for tc in self.tool_calls
                ]
            if self.invalid_tool_calls:
                tool_call_chunks = self.tool_call_chunks
                tool_call_chunks.extend(
                    [
                        create_tool_call_chunk(
                            name=tc["name"], args=tc["args"], id=tc["id"], index=None
                        )
                        for tc in self.invalid_tool_calls
                    ]
                )
                self.tool_call_chunks = tool_call_chunks

            return self
        tool_calls = []
        invalid_tool_calls = []

        def add_chunk_to_invalid_tool_calls(chunk: ToolCallChunk) -> None:
            invalid_tool_calls.append(
                create_invalid_tool_call(
                    name=chunk["name"],
                    args=chunk["args"],
                    id=chunk["id"],
                    error=None,
                )
            )

        for chunk in self.tool_call_chunks:
            try:
                args_ = parse_partial_json(chunk["args"]) if chunk["args"] else {}
                if isinstance(args_, dict):
                    tool_calls.append(
                        create_tool_call(
                            name=chunk["name"] or "",
                            args=args_,
                            id=chunk["id"],
                        )
                    )
                else:
                    add_chunk_to_invalid_tool_calls(chunk)
            except Exception:
                add_chunk_to_invalid_tool_calls(chunk)
        self.tool_calls = tool_calls
        self.invalid_tool_calls = invalid_tool_calls
        return self

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, AIMessageChunk):
            return add_ai_message_chunks(self, other)
        if isinstance(other, (list, tuple)) and all(
            isinstance(o, AIMessageChunk) for o in other
        ):
            return add_ai_message_chunks(self, *other)
        return super().__add__(other)


def add_ai_message_chunks(
    left: AIMessageChunk, *others: AIMessageChunk
) -> AIMessageChunk:
    """将多个 ``AIMessageChunk`` 对象相加。

    参数:
        left: 第一个 ``AIMessageChunk``。
        *others: 其他要相加的 ``AIMessageChunk``。

    异常:
        ValueError: 如果各个块的 example 值不相同。

    返回:
        相加后的 ``AIMessageChunk`` 结果。

    """
    if any(left.example != o.example for o in others):
        msg = "Cannot concatenate AIMessageChunks with different example values."
        raise ValueError(msg)

    content = merge_content(left.content, *(o.content for o in others))
    additional_kwargs = merge_dicts(
        left.additional_kwargs, *(o.additional_kwargs for o in others)
    )
    response_metadata = merge_dicts(
        left.response_metadata, *(o.response_metadata for o in others)
    )

    # Merge tool call chunks
    if raw_tool_calls := merge_lists(
        left.tool_call_chunks, *(o.tool_call_chunks for o in others)
    ):
        tool_call_chunks = [
            create_tool_call_chunk(
                name=rtc.get("name"),
                args=rtc.get("args"),
                index=rtc.get("index"),
                id=rtc.get("id"),
            )
            for rtc in raw_tool_calls
        ]
    else:
        tool_call_chunks = []

    # Token usage
    if left.usage_metadata or any(o.usage_metadata is not None for o in others):
        usage_metadata: Optional[UsageMetadata] = left.usage_metadata
        for other in others:
            usage_metadata = add_usage(usage_metadata, other.usage_metadata)
    else:
        usage_metadata = None

    chunk_id = None
    candidates = [left.id] + [o.id for o in others]
    # first pass: pick the first non-run-* id
    for id_ in candidates:
        if id_ and not id_.startswith(_LC_ID_PREFIX):
            chunk_id = id_
            break
    else:
        # second pass: no provider-assigned id found, just take the first non-null
        for id_ in candidates:
            if id_:
                chunk_id = id_
                break

    return left.__class__(
        example=left.example,
        content=content,
        additional_kwargs=additional_kwargs,
        tool_call_chunks=tool_call_chunks,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        id=chunk_id,
    )


def add_usage(
    left: Optional[UsageMetadata], right: Optional[UsageMetadata]
) -> UsageMetadata:
    """递归地将两个 UsageMetadata 对象相加。

    示例:
        .. code-block:: python

            from langchain_core.messages.ai import add_usage

            left = UsageMetadata(
                input_tokens=5,
                output_tokens=0,
                total_tokens=5,
                input_token_details=InputTokenDetails(cache_read=3),
            )
            right = UsageMetadata(
                input_tokens=0,
                output_tokens=10,
                total_tokens=10,
                output_token_details=OutputTokenDetails(reasoning=4),
            )

            add_usage(left, right)

        结果为

        .. code-block:: python

            UsageMetadata(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
                input_token_details=InputTokenDetails(cache_read=3),
                output_token_details=OutputTokenDetails(reasoning=4),
            )

    参数:
        left: 第一个 ``UsageMetadata`` 对象。
        right: 第二个 ``UsageMetadata`` 对象。

    返回:
        两个 ``UsageMetadata`` 对象的和。

    """
    if not (left or right):
        return UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
    if not (left and right):
        return cast("UsageMetadata", left or right)

    return UsageMetadata(
        **cast(
            "UsageMetadata",
            _dict_int_op(
                cast("dict", left),
                cast("dict", right),
                operator.add,
            ),
        )
    )


def subtract_usage(
    left: Optional[UsageMetadata], right: Optional[UsageMetadata]
) -> UsageMetadata:
    """递归地将两个 UsageMetadata 对象相减。

    Token 数量不能为负，所以实际操作是 max(left - right, 0)。

    示例:
        .. code-block:: python

            from langchain_core.messages.ai import subtract_usage

            left = UsageMetadata(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
                input_token_details=InputTokenDetails(cache_read=4),
            )
            right = UsageMetadata(
                input_tokens=3,
                output_tokens=8,
                total_tokens=11,
                output_token_details=OutputTokenDetails(reasoning=4),
            )

            subtract_usage(left, right)

        结果为

        .. code-block:: python

            UsageMetadata(
                input_tokens=2,
                output_tokens=2,
                total_tokens=4,
                input_token_details=InputTokenDetails(cache_read=4),
                output_token_details=OutputTokenDetails(reasoning=0),
            )

    参数:
        left: 第一个 ``UsageMetadata`` 对象。
        right: 第二个 ``UsageMetadata`` 对象。

    返回:
        相减后的 ``UsageMetadata`` 结果。

    """
    if not (left or right):
        return UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
    if not (left and right):
        return cast("UsageMetadata", left or right)

    return UsageMetadata(
        **cast(
            "UsageMetadata",
            _dict_int_op(
                cast("dict", left),
                cast("dict", right),
                (lambda le, ri: max(le - ri, 0)),
            ),
        )
    )
