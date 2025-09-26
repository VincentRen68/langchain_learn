"""
对比分析：只总结用户问题 vs 同时总结用户问题和AI回答
"""

from typing import Any, Optional
from langchain_core.memory import BaseMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


class UserOnlySummaryMemory(BaseMemory):
    """只对用户问题进行总结的记忆类"""
    
    llm: BaseLanguageModel
    memory_key: str = "user_summary"
    input_key: Optional[str] = None
    user_questions_buffer: str = ""
    
    # 只关注用户问题的摘要模板
    summary_prompt = PromptTemplate(
        input_variables=["summary", "new_questions"],
        template="""逐步总结用户的问题历史，在之前的摘要基础上进行补充，并返回一个新的摘要。

注意：只总结用户的问题和关注点，不要包含AI的回答内容。

当前摘要：
{summary}

新的用户问题：
{new_questions}

新摘要："""
    )
    
    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]
    
    def _get_input_key(self, inputs: dict[str, Any]) -> str:
        if self.input_key is None:
            memory_vars = set(self.memory_variables)
            input_keys = [k for k in inputs.keys() if k not in memory_vars and k != "stop"]
            if len(input_keys) != 1:
                raise ValueError(f"Expected one input key, got {input_keys}")
            return input_keys[0]
        return self.input_key
    
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {self.memory_key: self.user_questions_buffer}
    
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        # 只处理用户输入
        input_key = self._get_input_key(inputs)
        user_question = inputs[input_key]
        
        new_question_text = f"用户: {user_question}"
        chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
        self.user_questions_buffer = chain.predict(
            summary=self.user_questions_buffer,
            new_questions=new_question_text
        )
    
    def clear(self) -> None:
        self.user_questions_buffer = ""


class FullConversationSummaryMemory(BaseMemory):
    """同时总结用户问题和AI回答的记忆类"""
    
    llm: BaseLanguageModel
    memory_key: str = "conversation_summary"
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    conversation_buffer: str = ""
    
    # 总结完整对话的摘要模板
    summary_prompt = PromptTemplate(
        input_variables=["summary", "new_conversation"],
        template="""逐步总结所提供的对话内容，在之前的摘要基础上进行补充，并返回一个新的摘要。

注意：总结用户的问题和AI的回答，以及它们之间的关系。

当前摘要：
{summary}

新的对话内容：
{new_conversation}

新摘要："""
    )
    
    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]
    
    def _get_input_key(self, inputs: dict[str, Any]) -> str:
        if self.input_key is None:
            memory_vars = set(self.memory_variables)
            input_keys = [k for k in inputs.keys() if k not in memory_vars and k != "stop"]
            if len(input_keys) != 1:
                raise ValueError(f"Expected one input key, got {input_keys}")
            return input_keys[0]
        return self.input_key
    
    def _get_output_key(self, outputs: dict[str, str]) -> str:
        if len(outputs) == 1:
            return next(iter(outputs.keys()))
        elif "output" in outputs:
            return "output"
        else:
            raise ValueError(f"Expected one output key, got {outputs.keys()}")
    
    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {self.memory_key: self.conversation_buffer}
    
    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        # 处理用户输入和AI输出
        input_key = self._get_input_key(inputs)
        output_key = self._get_output_key(outputs)
        
        user_question = inputs[input_key]
        ai_answer = outputs[output_key]
        
        new_conversation_text = f"用户: {user_question}\nAI: {ai_answer}"
        chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)
        self.conversation_buffer = chain.predict(
            summary=self.conversation_buffer,
            new_conversation=new_conversation_text
        )
    
    def clear(self) -> None:
        self.conversation_buffer = ""


# 对比示例
def demonstrate_difference():
    """演示两种方式的区别"""
    
    print("=== 对话示例 ===")
    conversations = [
        ("什么是机器学习？", "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。"),
        ("深度学习有什么应用？", "深度学习在图像识别、语音识别、自然语言处理等领域有广泛应用。"),
        ("如何开始学习机器学习？", "建议从Python编程和数学基础开始，然后学习scikit-learn等工具。"),
        ("机器学习和深度学习有什么区别？", "机器学习是更广泛的概念，深度学习是机器学习的一个子集，使用神经网络。")
    ]
    
    print("\n=== 只总结用户问题的结果 ===")
    user_only_summary = """用户询问了关于机器学习的基本概念，包括机器学习的定义、深度学习的应用领域、学习机器学习的建议，以及机器学习和深度学习的区别。用户对人工智能技术很感兴趣，希望了解从基础概念到实际应用的全过程。"""
    
    print(user_only_summary)
    
    print("\n=== 同时总结用户问题和AI回答的结果 ===")
    full_conversation_summary = """用户询问了关于机器学习的基本概念。AI解释了机器学习是人工智能的分支，通过算法从数据中学习模式。用户进一步询问深度学习的应用，AI介绍了在图像识别、语音识别、自然语言处理等领域的应用。用户询问如何开始学习，AI建议从Python编程和数学基础开始，学习scikit-learn等工具。最后用户询问机器学习和深度学习的区别，AI解释深度学习是机器学习的子集，使用神经网络。整个对话展现了用户对AI技术的系统性学习需求。"""
    
    print(full_conversation_summary)
    
    print("\n=== 对比分析 ===")
    print("1. 信息完整性:")
    print("   - 只总结用户问题: 只记录用户关注点，缺少AI的回答内容")
    print("   - 完整对话总结: 包含问题和答案的完整信息")
    
    print("\n2. 上下文理解:")
    print("   - 只总结用户问题: 无法了解AI如何回答，缺少对话的完整性")
    print("   - 完整对话总结: 能够理解问答关系，提供更丰富的上下文")
    
    print("\n3. 隐私保护:")
    print("   - 只总结用户问题: 不保存AI回答，隐私保护更好")
    print("   - 完整对话总结: 保存AI回答，可能涉及敏感信息")
    
    print("\n4. 存储效率:")
    print("   - 只总结用户问题: 存储内容较少，效率更高")
    print("   - 完整对话总结: 存储内容较多，但信息更全面")
    
    print("\n5. 准确性:")
    print("   - 只总结用户问题: 对用户意图理解准确，但缺少答案验证")
    print("   - 完整对话总结: 对对话整体理解更准确，包含问答验证")


if __name__ == "__main__":
    demonstrate_difference()

