"""
自定义记忆类：只对用户提问进行总结，不对AI回答进行总结
"""

from typing import Any, Optional
from langchain_core.memory import BaseMemory
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


class UserOnlySummaryMemory(BaseMemory):
    """只对用户提问进行总结的记忆类。
    
    这个记忆类只保存和总结用户的提问，不保存AI的回答。
    适用于只需要了解用户关注点和问题历史的场景。
    """
    
    llm: BaseLanguageModel
    """用于生成摘要的语言模型。"""
    
    memory_key: str = "user_questions_summary"
    """记忆变量的键名。"""
    
    input_key: Optional[str] = None
    """输入键名。"""
    
    output_key: Optional[str] = None
    """输出键名。"""
    
    user_questions_buffer: str = ""
    """存储用户问题摘要的缓冲区。"""
    
    # 自定义的摘要提示模板，只关注用户问题
    summary_prompt = PromptTemplate(
        input_variables=["summary", "new_questions"],
        template="""逐步总结用户的问题历史，在之前的摘要基础上进行补充，并返回一个新的摘要。

注意：只总结用户的问题和关注点，不要包含AI的回答内容。

示例：
当前摘要：
用户询问了关于人工智能的基本概念和机器学习的基础知识。

新的用户问题：
用户: 深度学习有哪些应用领域？
用户: 如何开始学习深度学习？

新摘要：
用户询问了关于人工智能的基本概念和机器学习的基础知识。用户对深度学习很感兴趣，询问了深度学习的应用领域和学习方法。

示例结束

当前摘要：
{summary}

新的用户问题：
{new_questions}

新摘要："""
    )
    
    @property
    def memory_variables(self) -> list[str]:
        """返回记忆变量的列表。"""
        return [self.memory_key]
    
    def _get_input_key(self, inputs: dict[str, Any]) -> str:
        """获取输入键。"""
        if self.input_key is None:
            # 找到非记忆变量的输入键
            memory_vars = set(self.memory_variables)
            input_keys = [k for k in inputs.keys() if k not in memory_vars and k != "stop"]
            if len(input_keys) != 1:
                raise ValueError(f"Expected one input key, got {input_keys}")
            return input_keys[0]
        return self.input_key
    
    def _get_output_key(self, outputs: dict[str, str]) -> str:
        """获取输出键。"""
        if self.output_key is None:
            if len(outputs) == 1:
                return next(iter(outputs.keys()))
            elif "output" in outputs:
                return "output"
            else:
                raise ValueError(f"Expected one output key, got {outputs.keys()}")
        return self.output_key
    
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """加载记忆变量。"""
        return {self.memory_key: self.user_questions_buffer}
    
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """保存上下文，但只处理用户输入。"""
        # 只获取用户输入，忽略AI输出
        input_key = self._get_input_key(inputs)
        user_question = inputs[input_key]
        
        # 将用户问题格式化为字符串
        new_question_text = f"用户: {user_question}"
        
        # 生成新的摘要
        chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
        self.user_questions_buffer = chain.predict(
            summary=self.user_questions_buffer,
            new_questions=new_question_text
        )
    
    def clear(self) -> None:
        """清除记忆内容。"""
        self.user_questions_buffer = ""


class UserQuestionsOnlyMemory(BaseMemory):
    """只保存用户问题列表的记忆类。
    
    这个记忆类只保存用户的原始问题，不进行摘要处理。
    适用于需要保留完整问题历史的场景。
    """
    
    memory_key: str = "user_questions"
    """记忆变量的键名。"""
    
    input_key: Optional[str] = None
    """输入键名。"""
    
    user_questions: list[str] = []
    """存储用户问题的列表。"""
    
    max_questions: int = 10
    """最大保存的问题数量。"""
    
    @property
    def memory_variables(self) -> list[str]:
        """返回记忆变量的列表。"""
        return [self.memory_key]
    
    def _get_input_key(self, inputs: dict[str, Any]) -> str:
        """获取输入键。"""
        if self.input_key is None:
            memory_vars = set(self.memory_variables)
            input_keys = [k for k in inputs.keys() if k not in memory_vars and k != "stop"]
            if len(input_keys) != 1:
                raise ValueError(f"Expected one input key, got {input_keys}")
            return input_keys[0]
        return self.input_key
    
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """加载记忆变量。"""
        # 返回用户问题列表的字符串表示
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(self.user_questions)])
        return {self.memory_key: questions_text}
    
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """保存上下文，但只处理用户输入。"""
        # 只获取用户输入
        input_key = self._get_input_key(inputs)
        user_question = inputs[input_key]
        
        # 添加到问题列表
        self.user_questions.append(user_question)
        
        # 如果超过最大数量，移除最旧的问题
        if len(self.user_questions) > self.max_questions:
            self.user_questions.pop(0)
    
    def clear(self) -> None:
        """清除记忆内容。"""
        self.user_questions = []


# 使用示例
if __name__ == "__main__":
    from langchain.llms.fake import FakeLLM
    
    # 创建只总结用户问题的记忆
    memory = UserOnlySummaryMemory(
        llm=FakeLLM(),
        memory_key="user_summary"
    )
    
    # 模拟对话
    inputs1 = {"input": "什么是机器学习？"}
    outputs1 = {"output": "机器学习是人工智能的一个分支..."}
    
    inputs2 = {"input": "深度学习有什么应用？"}
    outputs2 = {"output": "深度学习在图像识别、自然语言处理等领域有广泛应用..."}
    
    # 保存上下文（只处理用户问题）
    memory.save_context(inputs1, outputs1)
    memory.save_context(inputs2, outputs2)
    
    # 加载记忆变量
    memory_vars = memory.load_memory_variables({})
    print("用户问题摘要:", memory_vars["user_summary"])

