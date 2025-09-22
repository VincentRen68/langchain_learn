from langchain_core.prompts.prompt import PromptTemplate

# 创建草稿回答的模板
_CREATE_DRAFT_ANSWER_TEMPLATE = """{question}\n\n"""
CREATE_DRAFT_ANSWER_PROMPT = PromptTemplate(
    input_variables=["question"], template=_CREATE_DRAFT_ANSWER_TEMPLATE
)

# 列出断言的模板
# 中文翻译：
# 这里有一个陈述：
# {statement}
# 列出一个项目符号列表，包含你在产生上述陈述时所做的假设。
_LIST_ASSERTIONS_TEMPLATE = """Here is a statement:
{statement}
Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""  # noqa: E501
LIST_ASSERTIONS_PROMPT = PromptTemplate(
    input_variables=["statement"], template=_LIST_ASSERTIONS_TEMPLATE
)

# 检查断言的模板
# 中文翻译：
# 这里有一个项目符号列表的断言：
# {assertions}
# 对于每个断言，确定它是真还是假。如果是假的，解释原因。
_CHECK_ASSERTIONS_TEMPLATE = """Here is a bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""  # noqa: E501
CHECK_ASSERTIONS_PROMPT = PromptTemplate(
    input_variables=["assertions"], template=_CHECK_ASSERTIONS_TEMPLATE
)

# 修改后的回答模板
# 中文翻译：
# {checked_assertions}
#
# 问题：根据上述断言和检查，你将如何回答问题'{question}'？
#
# 回答：
_REVISED_ANSWER_TEMPLATE = """{checked_assertions}

Question: In light of the above assertions and checks, how would you answer the question '{question}'?

Answer:"""  # noqa: E501
REVISED_ANSWER_PROMPT = PromptTemplate(
    input_variables=["checked_assertions", "question"],
    template=_REVISED_ANSWER_TEMPLATE,
)
