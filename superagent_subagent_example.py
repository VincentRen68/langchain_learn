#!/usr/bin/env python3
"""
LangChain SuperAgent调用SubAgent示例

本示例演示了如何使用LangChain的Runnable.as_tool()方法
将一个AgentExecutor转换为工具，从而让SuperAgent能够调用SubAgent。
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# 1. 创建SubAgent - 专门处理数学计算的代理
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """获取城市天气信息（模拟）"""
    weather_data = {
        "北京": "晴天，温度25°C",
        "上海": "多云，温度22°C", 
        "广州": "小雨，温度28°C"
    }
    return f"{city}的天气: {weather_data.get(city, '暂无数据')}"


def create_math_subagent():
    """创建专门处理数学的SubAgent"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 数学代理的工具
    math_tools = [calculate]
    
    # 创建数学代理的提示模板
    math_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的数学计算助手。
        你只能使用提供的计算工具来解决数学问题。
        请仔细分析问题，然后使用适当的工具进行计算。
        如果问题不是数学相关的，请说明你只能处理数学问题。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # 创建数学代理
    math_agent = create_react_agent(llm, math_tools, math_prompt)
    math_agent_executor = AgentExecutor(agent=math_agent, tools=math_tools, verbose=True)
    
    return math_agent_executor


def create_weather_subagent():
    """创建专门处理天气的SubAgent"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 天气代理的工具
    weather_tools = [get_weather]
    
    # 创建天气代理的提示模板
    weather_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的天气信息助手。
        你只能使用提供的天气工具来获取天气信息。
        请根据用户询问的城市，使用天气工具获取相关信息。
        如果用户询问的不是天气相关的问题，请说明你只能处理天气查询。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # 创建天气代理
    weather_agent = create_react_agent(llm, weather_tools, weather_prompt)
    weather_agent_executor = AgentExecutor(agent=weather_agent, tools=weather_tools, verbose=True)
    
    return weather_agent_executor


def create_superagent():
    """创建SuperAgent，能够调用SubAgent"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 创建SubAgent
    math_subagent = create_math_subagent()
    weather_subagent = create_weather_subagent()
    
    # 将SubAgent转换为工具
    math_tool = math_subagent.as_tool(
        name="math_expert",
        description="专门处理数学计算问题的专家代理。可以计算各种数学表达式。"
    )
    
    weather_tool = weather_subagent.as_tool(
        name="weather_expert", 
        description="专门处理天气查询问题的专家代理。可以查询指定城市的天气信息。"
    )
    
    # SuperAgent的工具列表（包含SubAgent转换的工具）
    superagent_tools = [math_tool, weather_tool]
    
    # 创建SuperAgent的提示模板
    superagent_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，能够协调不同的专家代理来解决问题。
        
        你有以下专家代理可以使用：
        1. math_expert: 专门处理数学计算问题
        2. weather_expert: 专门处理天气查询问题
        
        请根据用户的问题，选择合适的专家代理来处理。
        如果问题涉及多个领域，可以依次调用不同的专家代理。
        
        请用中文回答用户的问题。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # 创建SuperAgent
    superagent = create_react_agent(llm, superagent_tools, superagent_prompt)
    superagent_executor = AgentExecutor(agent=superagent, tools=superagent_tools, verbose=True)
    
    return superagent_executor


def main():
    """主函数 - 演示SuperAgent调用SubAgent"""
    print("=== LangChain SuperAgent调用SubAgent示例 ===\n")
    
    # 创建SuperAgent
    superagent = create_superagent()
    
    # 测试用例
    test_cases = [
        "请帮我计算 (25 + 17) * 3 的结果",
        "北京今天天气怎么样？",
        "请计算 100 / 4 的结果，然后告诉我上海的天气",
        "请帮我计算圆的面积，半径是5（公式：π * r²）"
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"测试 {i}: {question}")
        print("-" * 50)
        
        try:
            # 调用SuperAgent
            result = superagent.invoke({"input": question})
            print(f"SuperAgent回答: {result['output']}")
        except Exception as e:
            print(f"执行出错: {str(e)}")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
