# 说明（中文）：
# 系统消息前缀。用于告诉模型：尽可能好地回答问题，并告知其可用的工具。
# 英文模板含义："回答下面的问题，你可以使用以下工具。"
SYSTEM_MESSAGE_PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""  # noqa: E501
# 说明（中文）：
# 使用格式说明。指导模型如何以 JSON 结构给出工具调用，包括 action（工具名）与 action_input（工具输入）。
# 强调一次仅调用一个工具，并给出 Thought/Action/Observation/Final Answer 的完整交互格式。
FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""  # noqa: E501
# 说明（中文）：
# 系统消息后缀。提醒模型在给出最终答案时必须严格使用 `Final Answer` 这几个字符。
SYSTEM_MESSAGE_SUFFIX = """Begin! Reminder to always use the exact characters `Final Answer` when responding."""  # noqa: E501
# 说明（中文）：
# 人类消息模板。将用户输入与代理草稿本（agent_scratchpad，包含历史 Thought/Action/Observation）一并注入提示。
HUMAN_MESSAGE = "{input}\n\n{agent_scratchpad}"
