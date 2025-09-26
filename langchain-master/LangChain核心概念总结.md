### LangChain 核心概念：记忆与思考机制深度总结

本文档旨在全面总结 LangChain 框架中关于记忆、思考和规划的核心概念，并将这些概念与项目源代码中的具体功能模块进行关联，以便于深入理解其工作原理。

#### 一、草稿本 (Draft / Scratchpad)：实时思考的工作空间

*   **核心定义**:
    草稿本是 AI 模型在处理**单次**用户请求时，用于进行实时思考、推理、规划和决策的**临时工作空间**。它不是为了记录历史，而是为了完成当前任务。

*   **内容与格式**:
    它是模型的“内心独白”或“思考链”(Chain of Thought)，通常包含结构化的标签，如 `Thought:` (思考过程), `Action:` (行动计划), `Observation:` (工具返回的观察结果)。

*   **生命周期**:
    **短暂的 (Ephemeral)**。它只在一次请求的处理周期内存在于内存中，一旦生成最终答案，草稿本就会被**完全丢弃**。

*   **在项目中的位置**:
    1.  **代理框架 (Agent Framework)**: 这是草稿本最核心的应用。LangChain 的所有代理（Agent）执行器 (`AgentExecutor`) 的逻辑都围绕着这个内部的“Scratchpad”来构建。模型通过在草稿本上生成 `Thought/Action` 循环来决定如何调用工具。
    2.  **内容修正 (Content Correction)**: 在 `libs\langchain\langchain\chains\llm_checker\base.py` 文件中定义的 `LLMCheckerChain`，其内部使用的“草稿答案”(`create_draft_answer_chain`)就是草稿本概念的一个具体应用，用于对初步生成的内容进行迭代验证和修正。

#### 二、短期记忆 (Short-Term Memory)：无损的近期对话历史

*   **核心定义**:
    短期记忆是一个**缓冲区**，用于**完整、无损地记录**近期已完成的对话历史。

*   **内容与格式**:
    通常是**对话式**的，由一系列 `Human:` 和 `AI:` 的问答对组成，保留了交互的全部细节。

*   **生命周期**:
    **会话级的 (Session-based)**。它在整个对话会话中持续存在，并不断累积新的交互记录，直到达到某个限制。

*   **在项目中的位置**:
    *   `libs\langchain\langchain\memory\buffer.py`: 定义了 `ConversationBufferMemory`，这是最基础的短期记忆实现，会存储全部对话历史。
    *   `libs\langchain\langchain\memory\buffer_window.py`: 定义了 `ConversationBufferWindowMemory`，它只保留最近 `k` 次的对话，是短期记忆的一种长度限制策略。

#### 三、长期记忆 (Long-Term Memory)：精炼的持久化知识

*   **核心定义**:
    长期记忆是通过**压缩或索引**技术，对海量的、旧的对话历史进行**信息精炼**后形成的持久化知识库。

*   **内容与格式**:
    主要有两种形式：**摘要 (Summary)** 或 **向量化索引 (Vector Index)**。

*   **生命周期**:
    **持久的 (Persistent)**。它可以在整个会话，甚至跨会话中存在，代表了 AI 对过去交互的“核心理解”。

*   **在项目中的位置**:
    *   **摘要式**: `libs\langchain\langchain\memory\summary.py` 中定义的 `ConversationSummaryMemory`，它使用一个 LLM 将长对话总结成摘要。
    *   **检索式**: `libs\langchain\langchain\memory\vectorstore.py` 中定义的 `VectorStoreRetrieverMemory`，它将对话片段存入向量数据库，以便未来进行高效的语义检索。

#### 四、三者之间的交互关系

这是一个层级递进、分工明确的信息流：

1.  **记忆 → 草稿本**: 在新一轮请求开始时，**短期记忆和长期记忆**的内容被加载，作为上下文提供给模型，模型在此基础上开始**新的草稿本**进行思考。
2.  **草稿本 → 短期记忆**: 草稿本的**最终成果**（即最终答案）在与用户的原始问题配对后，被送入短期记忆进行完整记录。草稿本本身则被丢弃。
3.  **短期记忆 → 长期记忆**: 短期记忆中**最旧的部分**在达到长度阈值后，被**压缩**并转移到长期记忆中。

*   **在项目中的位置**:
    *   **短期 → 长期**: 这种交互的最佳范例是 `libs\langchain\langchain\memory\summary_buffer.py` 中定义的 `ConversationSummaryBufferMemory`。其内部的 `prune` 方法清晰地展示了当短期记忆缓冲区（buffer）过长时，如何将其中的旧内容压缩成摘要（summary）。
    *   **草稿 → 短期**: 这个关键转换发生在 `libs\langchain\langchain\agents\agent.py` 的 `AgentExecutor` 类的 `_call` 方法中。当代理的思考循环结束并得出最终答案（`AgentFinish`）后，执行器会调用记忆模块的 `save_context` 方法。

    ```python
    # 位于 libs\langchain\langchain\agents\agent.py -> AgentExecutor._call 方法内部

    # ... (代理执行思考、调用工具的循环) ...

    # 当循环结束，得到最终输出 output
    output = self._return(output, intermediate_steps, run_manager=run_manager)

    # 【关键步骤】如果配置了记忆模块
    if self.memory is not None:
        # 准备要存入记忆的输入和输出
        inputs = self.memory.load_memory_variables(inputs)
        # 调用 save_context 方法，将最终结果存入短期记忆
        self.memory.save_context(inputs, output)

    return output
    ```
    这段代码清晰地表明，只有在整个草稿本（思考链）的生命周期结束，并产出最终 `output` 之后，这个**最终结果**才会被存入记忆中。

#### 五、与不同思考模型的交互

草稿本是实现 ReAct、Plan-and-Execute 等高级代理架构的基础。

*   **ReAct (推理+行动)**: 模型的草稿本上会交替生成 `Thought:` (推理) 和 `Action:` (行动)，形成一个“思考-行动-观察”的循环。
*   **Plan and Execute (规划与执行)**: 模型首先在草稿本上制定一个详细的、多步骤的计划 (Plan)，然后执行器再逐一执行。
*   **Function Calling**: 现代 LLM 的函数调用功能是一种内置的 ReAct 形式。模型的“调用请求”就是一种结构化的**草稿 (`Action`)**。

*   **在项目中的位置**:
    所有这些思考模型都由 LangChain 的**代理模块 (`libs\langchain\langchain\agents`)** 支持和实现。`AgentExecutor` 的核心逻辑就是驱动模型生成草稿，并根据草稿内容（如 `Action`）来调用工具。

#### 六、压缩 (Compression) 步骤总结

*   **压缩对象**: **短期记忆**中**最旧**的部分。
*   **触发时机**: 在一次成功的"问-答"交互**完成并存入短期记忆之后**，系统会检查短期记忆的长度。如果超过阈值，则触发压缩。
*   **执行时间点**: **在两次大模型调用之间**，作为一种"内务管理"或"维护"任务。
*   **核心目的**: 为**下一次**大模型调用准备一个长度可控、信息丰富的上下文，防止因上下文过长导致的错误、高成本和性能下降。

#### 七、草稿本代码实现详解

基于本次会话的深入分析，以下是 LangChain 草稿本在源代码中的具体实现位置和功能：

##### 7.1 核心数据结构：intermediate_steps

**位置**: `libs/langchain/langchain/agents/agent.py` 第1587行
```python
intermediate_steps: list[tuple[AgentAction, str]] = []
```

**作用**: 这是草稿本的具体实现，记录代理的完整思考轨迹。每个元素是 `(AgentAction, observation)` 元组，包含：
- `AgentAction`: 动作信息（工具名、输入、日志等）
- `str`: 工具执行后的观察结果

**生命周期**:
1. **初始化**: 空列表 `[]`
2. **循环累积**: 每次LLM调用后添加新的 `(action, observation)` 对
3. **上下文传递**: 每次LLM调用时作为草稿本传递给模型
4. **最终输出**: 如果设置了 `return_intermediate_steps=True`，会包含在最终结果中
5. **丢弃**: 任务完成后，草稿本被丢弃（符合"短暂"特性）

##### 7.2 草稿本格式化工具

**位置**: `libs/langchain/langchain/agents/format_scratchpad/log.py`
```python
def format_log_to_str(
    intermediate_steps: list[tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
```

**作用**: 将 `intermediate_steps` 转换为字符串形式的草稿本，生成类似以下格式：
```
Thought: 我需要搜索信息
Action: search
Action Input: Python tutorial
Observation: 找到了相关教程
Thought: 
```

**使用场景**: ReAct、Structured Chat、Self-Ask 等需要字符串 scratchpad 的代理。

##### 7.3 草稿本模板类

**位置**: `libs/langchain/langchain/agents/schema.py`
```python
class AgentScratchPadChatPromptTemplate(ChatPromptTemplate):
    def _construct_agent_scratchpad(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
    ) -> str:
```

**作用**: 为聊天模型构建草稿本，添加上下文说明："This was your previous work (but I haven't seen any of it! I only see what you return as final answer)"

##### 7.4 代理执行器中的草稿本处理

**位置**: `libs/langchain/langchain/agents/agent.py` 的 `AgentExecutor` 类

**关键方法**:
- `_call()` (第1574行): 主要的执行循环，维护 `intermediate_steps` 列表
- `_iter_next_step()` (第1305行): 执行思考-动作-观察循环
- `_prepare_intermediate_steps()` (第1722行): 智能修剪中间步骤，控制上下文长度

**工作流程**:
```python
# 1. 初始化草稿本
intermediate_steps: list[tuple[AgentAction, str]] = []

# 2. 主循环
while self._should_continue(iterations, time_elapsed):
    # 3. 将当前草稿本传递给LLM
    next_step_output = self._take_next_step(..., intermediate_steps, ...)
    
    # 4. 将新的步骤添加到草稿本
    intermediate_steps.extend(next_step_output)
```

##### 7.5 草稿本在提示模板中的使用

**位置**: `libs/langchain/langchain/agents/chat/prompt.py`
```python
HUMAN_MESSAGE = "{input}\n\n{agent_scratchpad}"
```

**作用**: 将用户输入与草稿本（由 `intermediate_steps` 格式化而来）一起注入给模型，实现上下文传递。

##### 7.6 现代代理实现：Runnable 管道

**位置**: `libs/langchain/langchain/agents/structured_chat/base.py`
```python
def create_structured_chat_agent(...) -> Runnable:
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
        )
        | prompt
        | llm_with_stop
        | JSONAgentOutputParser()
    )
```

**作用**: 使用函数式管道处理草稿本，实现：
1. **统一接口**: 所有组件都遵循相同的调用模式
2. **管道式组合**: 使用 `|` 操作符轻松组合多个处理步骤
3. **声明式编程**: 描述"做什么"而不是"怎么做"
4. **内置功能**: 异步、流式、错误处理等开箱即用

##### 7.7 AgentAction 中的 log 参数

**位置**: `libs/core/langchain_core/agents.py`
```python
class AgentAction(Serializable):
    tool: str
    tool_input: Union[str, dict]
    log: str  # 存储LLM的完整原始输出
```

**作用**: `log` 参数存储了LLM的完整思考过程，包括：
- 思考过程（`Thought:`）
- 动作决策（`Action:`）
- 推理逻辑

**示例内容**:
```
Thought: 用户询问北京天气，我需要调用天气API获取信息。
Action: weather_api
Action Input: 北京
```

##### 7.8 草稿本与记忆系统的交互

**关键转换点**: `AgentExecutor._return()` 方法
```python
# 草稿本 → 短期记忆的转换
if self.memory is not None:
    inputs = self.memory.load_memory_variables(inputs)
    self.memory.save_context(inputs, output)  # 只有最终结果被保存
```

**重要特点**: 只有草稿本的**最终成果**（`AgentFinish`）才会被存入记忆，草稿本本身在任务完成后被完全丢弃。

##### 7.9 草稿本在不同代理类型中的应用

1. **ReAct 代理**: 使用字符串格式的草稿本
2. **Structured Chat 代理**: 使用 JSON 格式的工具调用
3. **Self-Ask with Search 代理**: 使用问答格式的草稿本
4. **OpenAI Tools 代理**: 使用函数调用格式

每种代理类型都有其特定的草稿本格式化方式，但核心概念都是相同的：**临时存储思考过程，为下一轮推理提供上下文**。

#### 八、SuperAgent调用SubAgent模式：双层记忆系统

基于本次会话的深入分析，LangChain支持SuperAgent调用已存在的SubAgent的功能，这种模式下的记忆系统与普通工具调用模式存在根本性差异。

##### 8.1 SuperAgent调用SubAgent的实现机制

**核心实现**: 通过 `Runnable.as_tool()` 方法将 `AgentExecutor` 转换为 `BaseTool`

**代码位置**: `libs/core/langchain_core/runnables/base.py:2512行`
```python
def as_tool(
    self,
    args_schema: Optional[type[BaseModel]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    arg_types: Optional[dict[str, type]] = None,
) -> BaseTool:
    """Create a BaseTool from a Runnable."""
```

**实现原理**:
1. `AgentExecutor` 继承自 `Chain`，而 `Chain` 继承自 `RunnableSerializable`
2. 因此 `AgentExecutor` 是一个 `Runnable` 对象
3. 通过 `as_tool()` 方法可以将其转换为工具，供其他代理调用

##### 8.2 双层记忆系统架构

**SuperAgent层记忆**:
- **草稿本**: 记录高层决策过程（调用哪个SubAgent）
- **短期记忆**: 记录与用户的完整对话历史
- **长期记忆**: 压缩SuperAgent层的对话历史

**SubAgent层记忆**:
- **草稿本**: 记录SubAgent内部的思考过程
- **短期记忆**: 记录SubAgent内部的交互历史
- **长期记忆**: 压缩SubAgent层的交互历史

**关键特点**:
- 两层记忆系统**完全隔离**，互不干扰
- SubAgent的内部记忆对SuperAgent**完全透明**
- SuperAgent只能看到SubAgent的最终输出

##### 8.3 记忆系统隔离机制对比

| 记忆层级 | SuperAgent调用SubAgent | Agent调用普通工具 |
|---------|----------------------|------------------|
| **草稿本** | **两层嵌套结构**<br/>- SuperAgent层：高层决策<br/>- SubAgent层：具体执行<br/>- SubAgent思考过程对SuperAgent透明 | **单层结构**<br/>- 所有思考过程在同一层级<br/>- 完整的推理链条可见<br/>- 直接的工具调用记录 |
| **短期记忆** | **两层独立系统**<br/>- SuperAgent：用户对话历史<br/>- SubAgent：内部交互历史<br/>- 完全隔离，避免信息泄露 | **单层统一系统**<br/>- 所有交互历史统一管理<br/>- 包含工具调用的完整记录<br/>- 简单直接的记忆管理 |
| **长期记忆** | **两层独立压缩**<br/>- 各自独立进行记忆压缩<br/>- 避免跨层记忆污染<br/>- 分层管理历史信息 | **单层统一压缩**<br/>- 统一进行记忆压缩<br/>- 包含所有交互的摘要<br/>- 简单直接的压缩策略 |

##### 8.4 记忆隔离的技术实现

**草稿本隔离**:
- **生命周期隔离**: 每次Agent调用时重新创建 `intermediate_steps = []`
- **作用域隔离**: 只在单次Agent执行周期内存在
- **数据结构隔离**: 使用独立的 `intermediate_steps` 列表存储
- **访问隔离**: 只有当前Agent可以访问自己的草稿本

**短期记忆隔离**:
- **实例隔离**: 每个AgentExecutor实例有独立的memory对象
- **会话隔离**: 在同一会话中持续累积
- **类型隔离**: 使用BaseChatMemory的不同实现
- **访问控制**: 通过 `memory.load_memory_variables()` 访问

**长期记忆隔离**:
- **压缩隔离**: 通过LLM将历史压缩为摘要
- **存储隔离**: 使用 `moving_summary_buffer` 独立存储
- **触发隔离**: 只在短期记忆超限时触发
- **持久隔离**: 可以跨会话保存和恢复

##### 8.5 记忆存储位置和唯一标识

**短期记忆存储**:
- **存储位置**: `InMemoryChatMessageHistory.messages`
- **数据结构**: `list[BaseMessage]`
- **唯一标识**: `BaseMessage.id` (可选字段，由LLM提供商生成)
- **代码位置**: `libs/core/langchain_core/chat_history.py:213行`

**长期记忆存储**:
- **存储位置**: `ConversationSummaryBufferMemory.moving_summary_buffer`
- **数据结构**: `str` (压缩后的摘要文本)
- **唯一标识**: 无直接唯一标识，通过实例引用
- **代码位置**: `libs/langchain/langchain/memory/summary_buffer.py:28行`

##### 8.6 记忆边界和转换机制

**草稿本 → 短期记忆**:
- **转换时机**: Agent调用结束时（AgentFinish）
- **隔离机制**: 完全隔离 - 只有最终输出被保存，草稿本被丢弃
- **代码位置**: `libs/langchain/langchain/chains/base.py:491行`

**短期记忆 → 长期记忆**:
- **转换时机**: 短期记忆超限时（prune触发）
- **隔离机制**: 压缩隔离 - 通过LLM压缩历史信息
- **代码位置**: `libs/langchain/langchain/memory/summary_buffer.py:114行`

**长期记忆 → 短期记忆**:
- **转换时机**: 每次Agent调用开始时
- **隔离机制**: 上下文隔离 - 摘要作为上下文提供
- **代码位置**: `libs/langchain/langchain/memory/summary_buffer.py:53行`

##### 8.7 设计考虑和性能影响

**设计考虑**:
- **SuperAgent模式**: 适合模块化设计，各SubAgent独立维护记忆
- **普通工具模式**: 适合统一管理，所有交互在同一记忆系统中
- **选择依据**: 取决于系统复杂度和模块化需求

**性能影响**:
- **SuperAgent模式**: 内存使用更高（多层记忆系统），但更好的模块化
- **普通工具模式**: 内存使用较低（单层记忆系统），管理复杂度更低

##### 8.8 实际应用示例

**创建SubAgent**:
```python
# 创建数学计算SubAgent
math_subagent = AgentExecutor(agent=math_agent, tools=math_tools)

# 创建天气查询SubAgent  
weather_subagent = AgentExecutor(agent=weather_agent, tools=weather_tools)
```

**转换为工具**:
```python
# 将SubAgent转换为工具
math_tool = math_subagent.as_tool(
    name="math_expert",
    description="专门处理数学计算问题的专家代理"
)

weather_tool = weather_subagent.as_tool(
    name="weather_expert", 
    description="专门处理天气查询问题的专家代理"
)
```

**SuperAgent使用SubAgent**:
```python
# SuperAgent的工具列表（包含SubAgent转换的工具）
superagent_tools = [math_tool, weather_tool]

# 创建SuperAgent
superagent = create_react_agent(llm, superagent_tools, prompt)
superagent_executor = AgentExecutor(agent=superagent, tools=superagent_tools)
```

这种双层记忆系统设计使得LangChain能够构建复杂的多代理系统，实现代理之间的协作和层次化管理，同时保持各层记忆系统的独立性和隔离性。
#### 八、长短期记忆代码位置详细索引

基于本次会话的深入分析，以下是 LangChain 框架中所有长短期记忆相关代码的详细位置索引：

##### 8.1 核心抽象层

**基础记忆抽象**:
- `libs/core/langchain_core/memory.py` - `BaseMemory` 抽象基类，定义记忆的核心接口
- `libs/langchain/langchain/memory/chat_memory.py` - `BaseChatMemory` 聊天记忆基类
- `libs/langchain/langchain/memory/utils.py` - 记忆工具函数

##### 8.2 短期记忆实现

**缓冲区记忆**:
- `libs/langchain/langchain/memory/buffer.py`
  - `ConversationBufferMemory` - 基础对话缓冲区记忆
  - `ConversationStringBufferMemory` - 字符串格式的缓冲区记忆（已弃用）

**窗口记忆**:
- `libs/langchain/langchain/memory/buffer_window.py`
  - `ConversationBufferWindowMemory` - 保留最近k轮对话的窗口记忆

**Token限制记忆**:
- `libs/langchain/langchain/memory/token_buffer.py`
  - `ConversationTokenBufferMemory` - 基于Token数量限制的记忆

##### 8.3 长期记忆实现

**摘要记忆**:
- `libs/langchain/langchain/memory/summary.py`
  - `ConversationSummaryMemory` - 持续总结对话历史的记忆
  - `SummarizerMixin` - 摘要器混入类
- `libs/langchain/langchain/memory/summary_buffer.py`
  - `ConversationSummaryBufferMemory` - 结合摘要和缓冲区的混合记忆

**向量存储记忆**:
- `libs/langchain/langchain/memory/vectorstore.py`
  - `VectorStoreRetrieverMemory` - 基于向量存储的对话历史检索记忆
- `libs/langchain/langchain/memory/vectorstore_token_buffer_memory.py`
  - `ConversationVectorStoreTokenBufferMemory` - 结合向量存储和Token缓冲区的记忆

**实体记忆**:
- `libs/langchain/langchain/memory/entity.py`
  - `ConversationEntityMemory` - 从对话中提取命名实体并生成摘要的记忆
  - `BaseEntityStore` - 实体存储抽象基类
  - `InMemoryEntityStore` - 内存实体存储
  - `RedisEntityStore` - Redis实体存储
  - `UpstashRedisEntityStore` - Upstash Redis实体存储
  - `SQLiteEntityStore` - SQLite实体存储

##### 8.4 存储后端实现

**聊天消息历史存储** (`libs/langchain/langchain/memory/chat_message_histories/`):

**内存存储**:
- `in_memory.py` - `InMemoryChatMessageHistory`

**数据库存储**:
- `redis.py` - `RedisChatMessageHistory`
- `mongodb.py` - `MongoDBChatMessageHistory`
- `postgres.py` - `PostgresChatMessageHistory`
- `sql.py` - `SQLChatMessageHistory`
- `singlestoredb.py` - `SingleStoreDBChatMessageHistory`

**云存储**:
- `astradb.py` - `AstraDBChatMessageHistory`
- `cosmos_db.py` - `CosmosDBChatMessageHistory`
- `dynamodb.py` - `DynamoDBChatMessageHistory`
- `elasticsearch.py` - `ElasticsearchChatMessageHistory`
- `firestore.py` - `FirestoreChatMessageHistory`
- `momento.py` - `MomentoChatMessageHistory`
- `neo4j.py` - `Neo4jChatMessageHistory`
- `rocksetdb.py` - `RocksetDBChatMessageHistory`
- `upstash_redis.py` - `UpstashRedisChatMessageHistory`
- `xata.py` - `XataChatMessageHistory`
- `zep.py` - `ZepChatMessageHistory`

**文件存储**:
- `file.py` - `FileChatMessageHistory`

**框架集成**:
- `streamlit.py` - `StreamlitChatMessageHistory`

##### 8.5 辅助组件

**组合记忆**:
- `libs/langchain/langchain/memory/combined.py` - `CombinedMemory` 组合多个记忆源

**简单记忆**:
- `libs/langchain/langchain/memory/simple.py` - `SimpleMemory` 简单的键值对记忆

**只读记忆**:
- `libs/langchain/langchain/memory/readonly.py` - `ReadOnlySharedMemory` 只读共享记忆

**知识图谱记忆**:
- `libs/langchain/langchain/memory/kg.py` - `ConversationKGMemory` 知识图谱记忆

**第三方记忆**:
- `libs/langchain/langchain/memory/motorhead_memory.py` - `MotorheadMemory`
- `libs/langchain/langchain/memory/zep_memory.py` - `ZepMemory`

**提示模板**:
- `libs/langchain/langchain/memory/prompt.py` - 记忆相关的提示模板

##### 8.6 文档和示例

**迁移指南** (`docs/docs/versions/migrating_memory/`):
- `index.mdx` - 记忆迁移指南总览
- `conversation_buffer_memory.ipynb` - 缓冲区记忆迁移示例
- `conversation_buffer_window_memory.ipynb` - 窗口记忆迁移示例
- `conversation_summary_memory.ipynb` - 摘要记忆迁移示例
- `chat_history.ipynb` - 聊天历史使用示例
- `long_term_memory_agent.ipynb` - 长期记忆智能体示例

**教程** (`docs/docs/tutorials/`):
- `qa_chat_history.ipynb` - 问答聊天历史
- `chatbot.ipynb` - 聊天机器人
- `summarization.ipynb` - 摘要功能

##### 8.7 测试文件

**记忆测试** (`libs/langchain/tests/unit_tests/memory/`):
- `test_combined_memory.py` - 组合记忆测试
- `test_imports.py` - 导入测试
- `chat_message_histories/test_imports.py` - 聊天历史导入测试

**记忆相关测试**:
- `libs/langchain/tests/unit_tests/chains/test_summary_buffer_memory.py` - 摘要缓冲区记忆测试
- `libs/langchain/tests/unit_tests/chains/test_memory.py` - 记忆测试
- `libs/langchain/tests/unit_tests/schema/test_memory.py` - 记忆模式测试

##### 8.8 重要说明

**弃用状态**:
- 大部分传统记忆抽象在 v0.3.1 版本已标记为弃用
- 计划在 v1.0.0 版本中移除
- 推荐迁移到 LangGraph 持久化功能

**迁移建议**:
1. **简单应用**: 继续使用 `RunnableWithMessageHistory` 和 `BaseChatMessageHistory`
2. **复杂应用**: 迁移到 LangGraph 持久化
3. **新项目**: 直接使用 LangGraph

**核心接口**:
- `load_memory_variables()` - 加载记忆变量
- `save_context()` - 保存上下文
- `clear()` - 清除记忆内容

