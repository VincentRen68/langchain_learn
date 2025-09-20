from langchain_core.prompts.prompt import PromptTemplate

_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = """你是一个由 OpenAI 训练的大型语言模型驱动的人类助手。

你被设计用来协助完成各种任务，从回答简单问题到就广泛主题提供深入的解释和讨论。作为一种语言模型，你能够根据收到的输入生成类似人类的文本，从而进行听起来自然的对话，并提供与当前主题连贯且相关的回应。

你在不断学习和进步，你的能力也在不断发展。你能够处理和理解大量文本，并可以利用这些知识为各种问题提供准确和信息丰富的回答。你可以访问下面“上下文”部分中由人类提供的一些个性化信息。此外，你能够根据收到的输入生成自己的文本，从而参与讨论并就广泛的主题提供解释和描述。

总的来说，你是一个强大的工具，可以帮助完成广泛的任务，并就各种主题提供有价值的见解和信息。无论人类是需要帮助解决特定问题，还是只想就某个特定主题进行对话，你都在这里提供协助。

上下文:
{entities}

当前对话:
{history}
最后一行:
人类: {input}
你:"""  # noqa: E501

ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
)

_DEFAULT_SUMMARIZER_TEMPLATE = """逐步总结所提供的对话内容，在之前的摘要基础上进行补充，并返回一个新的摘要。

示例
当前摘要:
人类询问 AI 对人工智能的看法。AI 认为人工智能是一股向善的力量。

新的对话内容:
人类: 为什么你认为人工智能是一股向善的力量？
AI: 因为人工智能将帮助人类充分发挥其潜力。

新摘要:
人类询问 AI 对人工智能的看法。AI 认为人工智能是一股向善的力量，因为它将帮助人类充分发挥其潜力。
示例结束

当前摘要:
{summary}

新的对话内容:
{new_lines}

新摘要:"""  # noqa: E501
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE
)

_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """你是一个正在阅读 AI 与人类对话记录的 AI 助手。从对话的最后一行中提取所有专有名词。作为指导，专有名词通常是首字母大写的。你必须提取所有的名字和地点。

提供对话历史记录是为了处理共指情况（例如，“你对他了解多少”中的“他”是在前文中定义的）——请忽略历史记录中提到但未出现在最后一行中的项目。

以逗号分隔的单个列表形式返回输出，如果没有任何值得注意的内容（例如，用户只是打个招呼或进行简单对话），则返回 NONE。

示例
对话历史:
人类 #1: 今天过得怎么样？
AI: "我过得很好！你呢？"
人类 #1: 不错！忙着在 Langchain 上工作，有很多事要做。
AI: "听起来工作量很大！你正在做什么来让 Langchain 变得更好？"
最后一行:
人类 #1: 我正在尝试改进 Langchain 的界面、用户体验、以及它与用户可能需要的各种产品的集成... 很多事情。
输出: Langchain
示例结束

示例
对话历史:
人类 #1: 今天过得怎么样？
AI: "我过得很好！你呢？"
人类 #1: 不错！忙着在 Langchain 上工作，有很多事要做。
AI: "听起来工作量很大！你正在做什么来让 Langchain 变得更好？"
最后一行:
人类 #1: 我正在尝试改进 Langchain 的界面、用户体验、以及它与用户可能需要的各种产品的集成... 很多事情。我正在和人类 #2 一起工作。
输出: Langchain, 人类 #2
示例结束

对话历史 (仅供参考):
{history}
最后一行对话 (用于提取):
人类: {input}

输出:"""  # noqa: E501
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)

_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE = """你是一个 AI 助手，正在帮助人类追踪他们生活中相关人物、地点和概念的事实。根据你与人类对话的最后一行，更新“实体”部分中提供的实体的摘要。如果是第一次编写摘要，请返回一个单句。
更新应仅包含对话最后一行中传达的关于所提供实体的事实，并且应仅包含关于该实体的事实。

如果关于所提供实体没有新信息，或者信息不值得注意（不是一个需要长期记住的重要或相关事实），请保持现有摘要不变并返回。

完整对话历史 (供参考):
{history}

要摘要的实体:
{entity}

{entity} 的现有摘要:
{summary}

最后一行对话:
人类: {input}
更新后的摘要:"""  # noqa: E501

ENTITY_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["entity", "summary", "history", "input"],
    template=_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE,
)


KG_TRIPLE_DELIMITER = "<|>"
_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "你是一个网络智能体，帮助人类追踪所有相关人物、事物、概念等的知识三元组，"
    "并将它们与存储在你权重中的知识以及知识图谱中存储的知识相集成。"
    "从对话的最后一行中提取所有的知识三元组。"
    "知识三元组是一个包含主语、谓语和宾语的子句。"
    "主语是被描述的实体，"
    "谓语是正在被描述的主语的属性，"
    "宾语是该属性的值。\n\n"
    "示例\n"
    "对话历史:\n"
    "人类 #1: 你听说外星人降落在 51 区了吗？\n"
    "AI: 没有，我没听说。你对 51 区了解多少？\n"
    "人类 #1: 这是内华达州的一个秘密军事基地。\n"
    "AI: 你对内华达州了解多少？\n"
    "最后一行对话:\n"
    "人类 #1: 它是美国的一个州。它也是美国第一大黄金生产地。\n\n"
    f"输出: (内华达州, 是一个, 州){KG_TRIPLE_DELIMITER}(内华达州, 位于, 美国)"
    f"{KG_TRIPLE_DELIMITER}(内华达州, 是第一大生产地, 黄金)\n"
    "示例结束\n\n"
    "示例\n"
    "对话历史:\n"
    "人类 #1: 你好。\n"
    "AI: 嗨！你好吗？\n"
    "人类 #1: 我很好。你呢？\n"
    "AI: 我也很好。\n"
    "最后一行对话:\n"
    "人类 #1: 我要去商店。\n\n"
    "输出: NONE\n"
    "示例结束\n\n"
    "示例\n"
    "对话历史:\n"
    "人类 #1: 你对笛卡尔了解多少？\n"
    "AI: 笛卡尔是 17 世纪的法国哲学家、数学家和科学家。\n"
    "人类 #1: 我指的是来自蒙特利尔的单口喜剧演员和室内设计师笛卡尔。\n"
    "AI: 哦是的，他是一名喜剧演员和室内设计师。他从事这个行业已经 30 年了。他最喜欢的食物是烤豆派。\n"
    "最后一行对话:\n"
    "人类 #1: 哦是吗。我知道笛卡尔喜欢开古董踏板车和弹曼陀林。\n"
    f"输出: (笛卡尔, 喜欢开, 古董踏板车){KG_TRIPLE_DELIMITER}(笛卡尔, 弹奏, 曼陀林)\n"
    "示例结束\n\n"
    "对话历史 (仅供参考):\n"
    "{history}"
    "\n最后一行对话 (用于提取):\n"
    "人类: {input}\n\n"
    "输出:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)
