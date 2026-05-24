from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langgraph.store.redis import RedisStore
from langgraph.checkpoint.redis import RedisSaver
from redis import Redis
from langchain_aws import ChatBedrockConverse
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from src.config.settings import settings
import boto3
import os


class BaseAgent(ABC):
    def __init__(
            self,
            name: str,
            model: Optional[ChatBedrockConverse] = None,
            system_prompt: Optional[str] = None,
    ):
        self.name = name
        memory_enabled = settings.enable_memory
        if memory_enabled:
            self.memory_enabled = True
        else:
            self.memory_enabled = False
        self.model = model or self._get_default_model()
        self.system_prompt = system_prompt or self._get_default_prompt()
        self.graph = None
        self.compiled_graph: Optional[CompiledStateGraph] = None
        self.connection_url: Optional[str] = None
        self.model_with_tools = None
        self.checkpointer = None
        self.store = None

    def _get_default_model(self):
        aws_region = settings.AWS_REGION_NAME
        LLM_id = settings.LLM_id
        temperature = settings.LLM_temperature
        LLM_max_tokens = settings.LLM_max_tokens
        bedrock_rt = boto3.client("bedrock-runtime", region_name=aws_region)
        self.model = ChatBedrockConverse(
            client=bedrock_rt,
            model=LLM_id,
            temperature=temperature,
            max_tokens=LLM_max_tokens,
        )
        return self.model

    def _get_default_prompt(self):
        return f"You are {self.name}, a helpful AI assistant."

    @abstractmethod
    def build_graph(self):
        pass

    def setup_storage(self):
        if not self.memory_enabled:
            return None

        redis_host = os.getenv("REDIS_HOST", "localhost")
        rq_conn = Redis(host=redis_host, socket_timeout=300)

        if self.checkpointer is None:
            self.checkpointer = RedisSaver(
                redis_client=rq_conn, ttl={"default_ttl": 60, "refresh_on_read": True}
            )
        if self.store is None:
            self.store = RedisStore(
                conn=rq_conn, ttl={"default_ttl": 60, "refresh_on_read": True}
            )
        self.checkpointer.setup()
        self.store.setup()

    def compile(self):
        if self.graph is None:
            self.graph = self.build_graph()

        compile_kwargs = {}
        if self.memory_enabled:
            self.setup_storage()
            compile_kwargs["checkpointer"] = self.checkpointer
            compile_kwargs["store"] = self.store

        self.compiled_graph = self.graph.compile(**compile_kwargs)
        return self.compiled_graph

    def get_state(self, thread_id: str):
        if not self.compiled_graph:
            raise RuntimeError("Graph is not compiled. Call compile() first.")
        config = {"configurable": {"thread_id": thread_id}}
        return self.compiled_graph.get_state(config)

    def update_state(self, thread_id: str, state_updates: Dict[str, Any]):
        if not self.compiled_graph:
            raise RuntimeError("Graph is not compiled. Call compile() first.")
        config = {"configurable": {"thread_id": thread_id}}
        self.compiled_graph.update_state(config, state_updates)

    async def close(self):
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None

    def invoke(
            self,
            inputs: Dict[str, Any],
            config: Optional[RunnableConfig] = None,
            thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.compiled_graph is None:
            self.compile()

        if config is None:
            config = {}

        if thread_id and self.memory_enabled:
            config["configurable"] = {"thread_id": thread_id}

        try:
            result = self.compiled_graph.invoke(inputs, config=config)
            return result
        except Exception as e:
            raise
