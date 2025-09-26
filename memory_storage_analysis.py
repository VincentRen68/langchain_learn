#!/usr/bin/env python3
"""
LangChain记忆系统存储位置和唯一标识分析

本文件详细分析短期记忆和长期记忆的存储位置、唯一标识符机制。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MemoryType(Enum):
    """记忆类型"""
    SHORT_TERM = "短期记忆"
    LONG_TERM = "长期记忆"


@dataclass
class MemoryStorage:
    """记忆存储信息"""
    memory_type: MemoryType
    storage_location: str
    unique_identifier: str
    data_structure: str
    code_location: str
    persistence_mechanism: str


def analyze_memory_storage():
    """
    分析记忆系统的存储位置和唯一标识
    """
    
    storage_analysis = {
        "短期记忆存储": {
            "存储位置": "内存中的Python列表",
            "具体位置": "InMemoryChatMessageHistory.messages",
            "数据结构": "list[BaseMessage]",
            "唯一标识": "BaseMessage.id (可选字段)",
            "代码位置": "libs/core/langchain_core/chat_history.py:213行",
            "关键代码": """
            class InMemoryChatMessageHistory(BaseChatMessageHistory, BaseModel):
                messages: list[BaseMessage] = Field(default_factory=list)
                \"\"\"A list of messages stored in memory.\"\"\"
            """,
            "存储机制": "内存存储，会话结束时丢失",
            "访问方式": "通过chat_memory.messages直接访问"
        },
        
        "长期记忆存储": {
            "存储位置": "内存中的字符串缓冲区",
            "具体位置": "ConversationSummaryBufferMemory.moving_summary_buffer",
            "数据结构": "str (压缩后的摘要文本)",
            "唯一标识": "无直接唯一标识，通过实例引用",
            "代码位置": "libs/langchain/langchain/memory/summary_buffer.py:28行",
            "关键代码": """
            class ConversationSummaryBufferMemory(BaseChatMemory, SummarizerMixin):
                moving_summary_buffer: str = ""
                \"\"\"移动摘要缓冲区，存储压缩后的历史摘要。\"\"\"
            """,
            "存储机制": "内存存储，可跨会话持久化",
            "访问方式": "通过moving_summary_buffer属性访问"
        }
    }
    
    return storage_analysis


def analyze_unique_identifiers():
    """
    分析记忆系统的唯一标识符机制
    """
    
    identifier_analysis = {
        "BaseMessage唯一标识": {
            "字段名": "id",
            "类型": "Optional[str]",
            "默认值": "None",
            "生成方式": "由创建消息的提供商/模型提供",
            "代码位置": "libs/core/langchain_core/messages/base.py:52行",
            "关键代码": """
            id: Optional[str] = Field(default=None, coerce_numbers_to_str=True)
            \"\"\"消息的可选唯一标识符。理想情况下，应由创建此消息的提供商/模型提供。\"\"\"
            """,
            "特点": [
                "可选字段，不是所有消息都有ID",
                "由外部系统（如LLM提供商）生成",
                "用于消息去重和追踪",
                "支持字符串和数字类型（自动转换为字符串）"
            ]
        },
        
        "记忆实例唯一标识": {
            "短期记忆": {
                "标识方式": "Python对象实例ID (id(chat_memory))",
                "唯一性": "每个AgentExecutor实例有独立的chat_memory对象",
                "生命周期": "与AgentExecutor实例绑定",
                "代码位置": "libs/langchain/langchain/memory/chat_memory.py:35行"
            },
            "长期记忆": {
                "标识方式": "Python对象实例ID (id(memory))",
                "唯一性": "每个ConversationSummaryBufferMemory实例独立",
                "生命周期": "与记忆实例绑定",
                "代码位置": "libs/langchain/langchain/memory/summary_buffer.py:20行"
            }
        }
    }
    
    return identifier_analysis


def analyze_storage_hierarchy():
    """
    分析存储层次结构
    """
    
    hierarchy = {
        "存储层次": {
            "Level 1 - 应用层": {
                "AgentExecutor": "管理整个代理的执行",
                "memory属性": "指向具体的记忆实现"
            },
            "Level 2 - 记忆层": {
                "ConversationSummaryBufferMemory": "长期记忆管理器",
                "BaseChatMemory": "短期记忆基类"
            },
            "Level 3 - 存储层": {
                "InMemoryChatMessageHistory": "短期记忆具体存储",
                "moving_summary_buffer": "长期记忆具体存储"
            },
            "Level 4 - 数据层": {
                "list[BaseMessage]": "短期记忆数据",
                "str": "长期记忆数据"
            }
        },
        
        "数据流": {
            "短期记忆": "用户输入 → BaseMessage → chat_memory.messages → list[BaseMessage]",
            "长期记忆": "短期记忆超限 → 压缩处理 → moving_summary_buffer → str"
        }
    }
    
    return hierarchy


def create_storage_diagram():
    """
    创建存储结构示意图
    """
    
    diagram = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                    LangChain记忆系统存储结构                        │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                        应用层 (Application Layer)                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────┐
    │  AgentExecutor  │
    │                 │
    │  memory: Memory │ ──────────┐
    └─────────────────┘           │
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        记忆层 (Memory Layer)                      │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────┐    ┌─────────────────────────────┐
    │ ConversationSummaryBuffer   │    │      BaseChatMemory         │
    │ Memory (长期记忆管理器)        │    │      (短期记忆基类)          │
    │                             │    │                             │
    │ moving_summary_buffer: str  │    │ chat_memory: ChatHistory    │
    └─────────────────────────────┘    └─────────────────────────────┘
                    │                                    │
                    │                                    │
                    ▼                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        存储层 (Storage Layer)                    │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────┐    ┌─────────────────────────────┐
    │    moving_summary_buffer    │    │ InMemoryChatMessageHistory  │
    │         (str)               │    │                             │
    │                             │    │ messages: list[BaseMessage] │
    │  "用户询问了数学问题，我调用   │    │                             │
    │   了计算工具得到结果..."      │    │ [HumanMessage, AIMessage,   │
    │                             │    │  HumanMessage, AIMessage]   │
    └─────────────────────────────┘    └─────────────────────────────┘
                    │                                    │
                    │                                    │
                    ▼                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        数据层 (Data Layer)                       │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────┐    ┌─────────────────────────────┐
    │         str                 │    │      BaseMessage            │
    │    (压缩摘要文本)             │    │                             │
    │                             │    │ id: Optional[str]          │
    │  • 无唯一标识符              │    │ content: str               │
    │  • 通过实例引用访问           │    │ type: str                  │
    │  • 跨会话持久化              │    │ additional_kwargs: dict    │
    │                             │    │ response_metadata: dict    │
    └─────────────────────────────┘    └─────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                        唯一标识符机制                            │
    └─────────────────────────────────────────────────────────────────┘
    
    短期记忆唯一标识：
    ├── 消息级别：BaseMessage.id (由LLM提供商生成)
    ├── 实例级别：id(chat_memory) (Python对象ID)
    └── 会话级别：AgentExecutor实例绑定
    
    长期记忆唯一标识：
    ├── 数据级别：无直接唯一标识
    ├── 实例级别：id(memory) (Python对象ID)
    └── 应用级别：AgentExecutor实例绑定
    """
    
    return diagram


def analyze_persistence_mechanisms():
    """
    分析持久化机制
    """
    
    persistence = {
        "短期记忆持久化": {
            "默认机制": "内存存储 (InMemoryChatMessageHistory)",
            "持久化选项": [
                "PostgresChatMessageHistory - 数据库存储",
                "RedisChatMessageHistory - Redis存储", 
                "DynamoDBChatMessageHistory - DynamoDB存储",
                "MongoDBChatMessageHistory - MongoDB存储"
            ],
            "唯一标识": "通过数据库主键或Redis键",
            "代码位置": "libs/core/langchain_core/chat_history.py"
        },
        
        "长期记忆持久化": {
            "默认机制": "内存存储 (moving_summary_buffer)",
            "持久化选项": [
                "自定义存储后端",
                "文件系统存储",
                "数据库存储"
            ],
            "唯一标识": "通过存储键或文件路径",
            "代码位置": "libs/langchain/langchain/memory/summary_buffer.py"
        }
    }
    
    return persistence


def main():
    """主函数"""
    print("=" * 80)
    print("LangChain记忆系统存储位置和唯一标识分析")
    print("=" * 80)
    
    # 获取分析结果
    storage_analysis = analyze_memory_storage()
    identifier_analysis = analyze_unique_identifiers()
    hierarchy = analyze_storage_hierarchy()
    diagram = create_storage_diagram()
    persistence = analyze_persistence_mechanisms()
    
    # 打印存储分析
    print("\n## 记忆系统存储位置")
    print("-" * 50)
    
    for memory_type, details in storage_analysis.items():
        print(f"\n### {memory_type}")
        print(f"**存储位置:** {details['存储位置']}")
        print(f"**具体位置:** {details['具体位置']}")
        print(f"**数据结构:** {details['数据结构']}")
        print(f"**唯一标识:** {details['唯一标识']}")
        print(f"**代码位置:** {details['代码位置']}")
        print(f"**关键代码:**")
        print(details['关键代码'])
        print(f"**存储机制:** {details['存储机制']}")
        print(f"**访问方式:** {details['访问方式']}")
    
    # 打印唯一标识符分析
    print(f"\n## 唯一标识符机制")
    print("-" * 50)
    
    for identifier_type, details in identifier_analysis.items():
        print(f"\n### {identifier_type}")
        if isinstance(details, dict) and '字段名' in details:
            print(f"**字段名:** {details['字段名']}")
            print(f"**类型:** {details['类型']}")
            print(f"**默认值:** {details['默认值']}")
            print(f"**生成方式:** {details['生成方式']}")
            print(f"**代码位置:** {details['代码位置']}")
            print(f"**关键代码:**")
            print(details['关键代码'])
            print(f"**特点:**")
            for feature in details['特点']:
                print(f"  - {feature}")
        else:
            for sub_type, sub_details in details.items():
                print(f"\n**{sub_type}:**")
                for key, value in sub_details.items():
                    print(f"  - {key}: {value}")
    
    # 打印存储层次结构
    print(f"\n## 存储层次结构")
    print("-" * 50)
    
    for level, components in hierarchy['存储层次'].items():
        print(f"\n### {level}")
        for component, description in components.items():
            print(f"  - {component}: {description}")
    
    print(f"\n### 数据流")
    for flow_type, flow_description in hierarchy['数据流'].items():
        print(f"  - {flow_type}: {flow_description}")
    
    # 打印持久化机制
    print(f"\n## 持久化机制")
    print("-" * 50)
    
    for memory_type, details in persistence.items():
        print(f"\n### {memory_type}")
        print(f"**默认机制:** {details['默认机制']}")
        print(f"**持久化选项:**")
        for option in details['持久化选项']:
            print(f"  - {option}")
        print(f"**唯一标识:** {details['唯一标识']}")
        print(f"**代码位置:** {details['代码位置']}")
    
    # 打印存储结构示意图
    print(f"\n## 存储结构示意图")
    print("-" * 50)
    print(diagram)


if __name__ == "__main__":
    main()
