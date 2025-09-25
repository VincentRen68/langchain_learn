from langchain_core.agents import AgentAction


def format_log_to_str(
    intermediate_steps: list[tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """构建草稿本，让代理能够继续其思考过程。

    参数:
        intermediate_steps: AgentAction 和观察结果字符串的元组列表。
        observation_prefix: 附加到观察结果的前缀。
             默认为 "Observation: "。
        llm_prefix: 附加到 LLM 调用的前缀。
               默认为 "Thought: "。

    返回:
        str: 草稿本。
    """
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts
