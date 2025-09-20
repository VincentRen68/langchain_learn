"""自 LangChain v0.3.4 起已弃用，并将在 LangChain v1.0.0 中移除。"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING, Any, Optional

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts import BasePromptTemplate
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_SUMMARIZATION_PROMPT,
)
from langchain.memory.utils import get_prompt_input_key

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class BaseEntityStore(BaseModel, ABC):
    """实体存储的抽象基类。"""

    @abstractmethod
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """从存储中获取实体值。"""

    @abstractmethod
    def set(self, key: str, value: Optional[str]) -> None:
        """在存储中设置实体值。"""

    @abstractmethod
    def delete(self, key: str) -> None:
        """从存储中删除实体值。"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查实体是否存在于存储中。"""

    @abstractmethod
    def clear(self) -> None:
        """从存储中删除所有实体。"""


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class InMemoryEntityStore(BaseEntityStore):
    """内存中的实体存储。"""

    store: dict[str, Optional[str]] = {}

    @override
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.store.get(key, default)

    @override
    def set(self, key: str, value: Optional[str]) -> None:
        self.store[key] = value

    @override
    def delete(self, key: str) -> None:
        del self.store[key]

    @override
    def exists(self, key: str) -> bool:
        return key in self.store

    @override
    def clear(self) -> None:
        return self.store.clear()


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class UpstashRedisEntityStore(BaseEntityStore):
    """由 Upstash Redis 支持的实体存储。

    实体默认的 TTL（生存时间）为 1 天，
    每次读取实体时，其 TTL 会延长 3 天。
    """

    def __init__(
        self,
        session_id: str = "default",
        url: str = "",
        token: str = "",
        key_prefix: str = "memory_store",
        ttl: Optional[int] = 60 * 60 * 24,
        recall_ttl: Optional[int] = 60 * 60 * 24 * 3,
        *args: Any,
        **kwargs: Any,
    ):
        """初始化 RedisEntityStore。

        参数:
            session_id: 会话的唯一标识符。
            url: Redis 服务器的 URL。
            token: Redis 服务器的认证令牌。
            key_prefix: Redis 存储中键的前缀。
            ttl: 键的生存时间（秒），默认为 1 天。
            recall_ttl: 键被重新调用时延长的时间（秒），默认为 3 天。
            *args: 额外的 positional 参数。
            **kwargs: 额外的 keyword 参数。
        """
        try:
            from upstash_redis import Redis
        except ImportError as e:
            msg = (
                "Could not import upstash_redis python package. "
                "Please install it with `pip install upstash_redis`."
            )
            raise ImportError(msg) from e

        super().__init__(*args, **kwargs)

        try:
            self.redis_client = Redis(url=url, token=token)
        except Exception as exc:
            error_msg = "Upstash Redis instance could not be initiated"
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from exc

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.recall_ttl = recall_ttl or ttl

    @property
    def full_key_prefix(self) -> str:
        """返回带有会话 ID 的完整键前缀。"""
        return f"{self.key_prefix}:{self.session_id}"

    @override
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        res = (
            self.redis_client.getex(f"{self.full_key_prefix}:{key}", ex=self.recall_ttl)
            or default
            or ""
        )
        logger.debug(
            "Upstash Redis MEM get '%s:%s': '%s'", self.full_key_prefix, key, res
        )
        return res

    @override
    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        self.redis_client.set(f"{self.full_key_prefix}:{key}", value, ex=self.ttl)
        logger.debug(
            "Redis MEM set '%s:%s': '%s' EX %s",
            self.full_key_prefix,
            key,
            value,
            self.ttl,
        )
        return None

    @override
    def delete(self, key: str) -> None:
        self.redis_client.delete(f"{self.full_key_prefix}:{key}")

    @override
    def exists(self, key: str) -> bool:
        return self.redis_client.exists(f"{self.full_key_prefix}:{key}") == 1

    @override
    def clear(self) -> None:
        def scan_and_delete(cursor: int) -> int:
            cursor, keys_to_delete = self.redis_client.scan(
                cursor,
                f"{self.full_key_prefix}:*",
            )
            self.redis_client.delete(*keys_to_delete)
            return cursor

        cursor = scan_and_delete(0)
        while cursor != 0:
            scan_and_delete(cursor)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class RedisEntityStore(BaseEntityStore):
    """由 Redis 支持的实体存储。

    实体默认的 TTL（生存时间）为 1 天，
    每次读取实体时，其 TTL 会延长 3 天。
    """

    redis_client: Any
    session_id: str = "default"
    key_prefix: str = "memory_store"
    ttl: Optional[int] = 60 * 60 * 24
    recall_ttl: Optional[int] = 60 * 60 * 24 * 3

    def __init__(
        self,
        session_id: str = "default",
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "memory_store",
        ttl: Optional[int] = 60 * 60 * 24,
        recall_ttl: Optional[int] = 60 * 60 * 24 * 3,
        *args: Any,
        **kwargs: Any,
    ):
        """初始化 RedisEntityStore。

        参数:
            session_id: 会话的唯一标识符。
            url: Redis 服务器的 URL。
            key_prefix: Redis 存储中键的前缀。
            ttl: 键的生存时间（秒），默认为 1 天。
            recall_ttl: 键被重新调用时延长的时间（秒），默认为 3 天。
            *args: 额外的 positional 参数。
            **kwargs: 额外的 keyword 参数。
        """
        try:
            import redis
        except ImportError as e:
            msg = (
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
            raise ImportError(msg) from e

        super().__init__(*args, **kwargs)

        try:
            from langchain_community.utilities.redis import get_client
        except ImportError as e:
            msg = (
                "Could not import langchain_community.utilities.redis.get_client. "
                "Please install it with `pip install langchain-community`."
            )
            raise ImportError(msg) from e

        try:
            self.redis_client = get_client(redis_url=url, decode_responses=True)
        except redis.exceptions.ConnectionError:
            logger.exception("Redis client could not connect")

        self.session_id = session_id
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.recall_ttl = recall_ttl or ttl

    @property
    def full_key_prefix(self) -> str:
        """返回带有会话 ID 的完整键前缀。"""
        return f"{self.key_prefix}:{self.session_id}"

    @override
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        res = (
            self.redis_client.getex(f"{self.full_key_prefix}:{key}", ex=self.recall_ttl)
            or default
            or ""
        )
        logger.debug("REDIS MEM get '%s:%s': '%s'", self.full_key_prefix, key, res)
        return res

    @override
    def set(self, key: str, value: Optional[str]) -> None:
        if not value:
            return self.delete(key)
        self.redis_client.set(f"{self.full_key_prefix}:{key}", value, ex=self.ttl)
        logger.debug(
            "REDIS MEM set '%s:%s': '%s' EX %s",
            self.full_key_prefix,
            key,
            value,
            self.ttl,
        )
        return None

    @override
    def delete(self, key: str) -> None:
        self.redis_client.delete(f"{self.full_key_prefix}:{key}")

    @override
    def exists(self, key: str) -> bool:
        return self.redis_client.exists(f"{self.full_key_prefix}:{key}") == 1

    @override
    def clear(self) -> None:
        # 以 batch_size 为大小，分批迭代列表
        def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[Any]:
            iterator = iter(iterable)
            while batch := list(islice(iterator, batch_size)):
                yield batch

        for keybatch in batched(
            self.redis_client.scan_iter(f"{self.full_key_prefix}:*"),
            500,
        ):
            self.redis_client.delete(*keybatch)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class SQLiteEntityStore(BaseEntityStore):
    """使用安全查询构造的 SQLite 支持的实体存储。"""

    session_id: str = "default"
    table_name: str = "memory_store"
    conn: Any = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(
        self,
        session_id: str = "default",
        db_file: str = "entities.db",
        table_name: str = "memory_store",
        *args: Any,
        **kwargs: Any,
    ):
        """初始化 SQLiteEntityStore。

        参数:
            session_id: 会话的唯一标识符。
            db_file: SQLite 数据库文件的路径。
            table_name: 用于存储实体的表的名称。
            *args: 额外的 positional 参数。
            **kwargs: 额外的 keyword 参数。
        """
        super().__init__(*args, **kwargs)
        try:
            import sqlite3
        except ImportError as e:
            msg = (
                "Could not import sqlite3 python package. "
                "Please install it with `pip install sqlite3`."
            )
            raise ImportError(msg) from e

        # 基本验证，以防止明显的恶意表名/会话名
        if not table_name.isidentifier() or not session_id.isidentifier():
            # 由于我们在此处进行了验证，因此可以安全地抑制 S608 bandit 警告
            msg = "Table name and session ID must be valid Python identifiers."
            raise ValueError(msg)

        self.conn = sqlite3.connect(db_file)
        self.session_id = session_id
        self.table_name = table_name
        self._create_table_if_not_exists()

    @property
    def full_table_name(self) -> str:
        """返回带有会话 ID 的完整表名。"""
        return f"{self.table_name}_{self.session_id}"

    def _execute_query(self, query: str, params: tuple = ()) -> "sqlite3.Cursor":
        """使用正确的连接处理方式执行查询。"""
        with self.conn:
            return self.conn.execute(query, params)

    def _create_table_if_not_exists(self) -> None:
        """如果实体表不存在，则使用安全的引用方式创建它。"""
        # 对表名标识符使用标准的 SQL 双引号
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS "{self.full_table_name}" (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        self._execute_query(create_table_query)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """安全地引用表名来检索值。"""
        # 使用 `?` 占位符来防止 SQL 注入
        # 忽略 S608，因为我们在 `__init__` 中验证了恶意的表名/会话名
        query = f'SELECT value FROM "{self.full_table_name}" WHERE key = ?'  # noqa: S608
        cursor = self._execute_query(query, (key,))
        result = cursor.fetchone()
        return result[0] if result is not None else default

    def set(self, key: str, value: Optional[str]) -> None:
        """安全地引用表名来插入或替换值。"""
        if not value:
            return self.delete(key)
        # 忽略 S608，因为我们在 `__init__` 中验证了恶意的表名/会话名
        query = (
            "INSERT OR REPLACE INTO "  # noqa: S608
            f'"{self.full_table_name}" (key, value) VALUES (?, ?)'
        )
        self._execute_query(query, (key, value))
        return None

    def delete(self, key: str) -> None:
        """安全地引用表名来删除键值对。"""
        # 忽略 S608，因为我们在 `__init__` 中验证了恶意的表名/会话名
        query = f'DELETE FROM "{self.full_table_name}" WHERE key = ?'  # noqa: S608
        self._execute_query(query, (key,))

    def exists(self, key: str) -> bool:
        """安全地引用表名来检查键是否存在。"""
        # 忽略 S608，因为我们在 `__init__` 中验证了恶意的表名/会话名
        query = f'SELECT 1 FROM "{self.full_table_name}" WHERE key = ? LIMIT 1'  # noqa: S608
        cursor = self._execute_query(query, (key,))
        return cursor.fetchone() is not None

    @override
    def clear(self) -> None:
        # 忽略 S608，因为我们在 `__init__` 中验证了恶意的表名/会话名
        query = f"""
            DELETE FROM {self.full_table_name}
        """  # noqa: S608
        with self.conn:
            self.conn.execute(query)


@deprecated(
    since="0.3.1",
    removal="1.0.0",
    message=(
        "Please see the migration guide at: "
        "https://python.langchain.com/docs/versions/migrating_memory/"
    ),
)
class ConversationEntityMemory(BaseChatMemory):
    """实体提取与摘要记忆。

    从最近的聊天历史中提取命名实体并生成摘要。
    通过可替换的实体存储，在不同对话间持久化实体。
    默认为内存实体存储，可以替换为 Redis、SQLite 或其他实体存储。
    """

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    entity_summarization_prompt: BasePromptTemplate = ENTITY_SUMMARIZATION_PROMPT

    # 最近检测到的实体名称缓存（如果有的话）
    # 在调用 load_memory_variables 时更新：
    entity_cache: list[str] = []

    # 更新实体时要考虑的最近消息对的数量：
    k: int = 3

    chat_history_key: str = "history"

    # 用于管理实体相关数据的存储：
    entity_store: BaseEntityStore = Field(default_factory=InMemoryEntityStore)

    @property
    def buffer(self) -> list[BaseMessage]:
        """访问聊天记忆消息。"""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> list[str]:
        """总是返回内存变量的列表。

        :meta private:
        """
        return ["entities", self.chat_history_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """加载内存变量。

        返回聊天历史和所有已生成的实体及其摘要（如果可用），
        并更新或清除最近的实体缓存。

        在调用此方法时，可能会在生成实体摘要之前发现新的实体名称，
        因此，如果尚未生成实体描述，则实体缓存值可能为空。
        """
        # 创建一个 LLMChain，用于从最近的聊天历史中预测实体名称：
        chain = LLMChain(llm=self.llm, prompt=self.entity_extraction_prompt)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        # 从聊天历史中提取最后 k 对消息的任意窗口，
        # 其中超参数 k 是消息对的数量：
        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        # 生成一个逗号分隔的命名实体列表，
        # 例如 "Jane, White House, UFO"
        # 如果没有提取到命名实体，则为 "NONE"：
        output = chain.predict(
            history=buffer_string,
            input=inputs[prompt_input_key],
        )

        # 如果没有提取到命名实体，则分配一个空列表。
        if output.strip() == "NONE":
            entities = []
        else:
            # 创建一个提取出的实体的列表：
            entities = [w.strip() for w in output.split(",")]

        # 创建一个包含实体及其摘要（如果存在）的字典：
        entity_summaries = {}

        for entity in entities:
            entity_summaries[entity] = self.entity_store.get(entity, "")

        # 将实体名称缓存替换为最近讨论的实体，
        # 如果没有提取到实体，则清除缓存：
        self.entity_cache = entities

        # 我们应该以消息对象的形式返回还是以字符串的形式返回？
        if self.return_messages:
            # 获取最后 `k` 对聊天消息：
            buffer: Any = self.buffer[-self.k * 2 :]
        else:
            # 重用我们之前创建的字符串：
            buffer = buffer_string

        return {
            self.chat_history_key: buffer,
            "entities": entity_summaries,
        }

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """将本次对话历史的上下文保存到实体存储中。

        通过提示模型为实体缓存中的每个实体生成摘要，
        并将这些摘要保存到实体存储中。
        """
        super().save_context(inputs, outputs)

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        # 从聊天历史中提取最后 k 对消息的任意窗口，
        # 其中超参数 k 是消息对的数量：
        buffer_string = get_buffer_string(
            self.buffer[-self.k * 2 :],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        input_data = inputs[prompt_input_key]

        # 创建一个 LLMChain，用于从上下文中预测实体摘要
        chain = LLMChain(llm=self.llm, prompt=self.entity_summarization_prompt)

        # 为实体生成新的摘要并将其保存在实体存储中
        for entity in self.entity_cache:
            # 如果存在，则获取现有摘要
            existing_summary = self.entity_store.get(entity, "")
            output = chain.predict(
                summary=existing_summary,
                entity=entity,
                history=buffer_string,
                input=input_data,
            )
            # 将更新后的摘要保存到实体存储中
            self.entity_store.set(entity, output.strip())

    def clear(self) -> None:
        """清除记忆内容。"""
        self.chat_memory.clear()
        self.entity_cache.clear()
        self.entity_store.clear()
