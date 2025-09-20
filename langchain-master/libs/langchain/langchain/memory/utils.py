from typing import Any


def get_prompt_input_key(inputs: dict[str, Any], memory_variables: list[str]) -> str:
    """获取提示的输入键。

    参数:
        inputs: Dict[str, Any]
        memory_variables: List[str]

    返回:
        一个提示输入键。
    """
    # "stop" 是一个特殊的键，可以作为输入传递，但不会用于格式化提示。
    prompt_input_keys = list(set(inputs).difference([*memory_variables, "stop"]))
    if len(prompt_input_keys) != 1:
        msg = f"One input key expected got {prompt_input_keys}"
        raise ValueError(msg)
    return prompt_input_keys[0]
