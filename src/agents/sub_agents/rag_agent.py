import json
from typing import Dict, Any, Literal

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, END

from src.core.base_agent import BaseAgent
from src.core.state import RAGAgentState, InputState, OutputState
from src.agents.tools.qdrant_tools import qdrant_search_tool
from src.utils.loggers import logger

_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions using retrieved context.

Instructions:
1. Call the `qdrant_search` tool with the user's question to retrieve relevant document chunks.
2. Once you have the retrieved chunks, synthesise a clear, accurate answer grounded in that context.
3. If the retrieved context does not contain enough information, say so explicitly rather than guessing.
4. Do not fabricate information not present in the retrieved chunks.

Tone guidelines based on detected user emotion:
- frustrated / concerned / anxious → calm, direct, solution-focused
- happy                            → warm, positive
- neutral                          → professional and precise
"""


class RAGAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="RAG Agent", system_prompt=_SYSTEM_PROMPT, **kwargs)
        _tools = [qdrant_search_tool]
        self.tools_by_name = {t.name: t for t in _tools}
        self.model_with_tools = self.model.bind_tools(_tools)

    def retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emotion = state.get("emotion") or "neutral"
        system_content = (
            self.system_prompt
            + f"\n\n[Detected user emotion: '{emotion}'. Adjust response tone accordingly.]"
        )
        messages = [SystemMessage(content=system_content)]
        messages += trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=12_000,
            start_on="human",
            end_on=("human", "tool"),
        )
        response = self.model_with_tools.invoke(messages)
        logger.info(
            "RAGAgent.retrieve_node: tool_calls=%d emotion=%s",
            len(getattr(response, "tool_calls", [])),
            emotion,
        )
        return {"messages": [response]}

    def tool_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        for tc in state["messages"][-1].tool_calls:
            fn = self.tools_by_name.get(tc["name"])
            if fn is None:
                content = json.dumps({"error": f"Tool '{tc['name']}' not available"})
            else:
                try:
                    observation = fn.invoke(tc["args"])
                    content = (
                        json.dumps(observation)
                        if not isinstance(observation, str)
                        else observation
                    )
                    logger.debug(
                        "RAGAgent tool '%s' returned %d chunk(s)",
                        tc["name"],
                        len(observation) if isinstance(observation, list) else 1,
                    )
                except Exception as e:
                    logger.error("RAGAgent tool '%s' error: %s", tc["name"], e, exc_info=True)
                    content = json.dumps({"error": str(e)})
            results.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        return {"messages": results}

    def should_continue(self, state: Dict[str, Any]) -> Literal["tool_node", "__end__"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tool_node"
        return END

    def build_graph(self):
        graph = StateGraph(RAGAgentState, input=InputState, output=OutputState)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("tool_node", self.tool_node)
        graph.add_edge(START, "retrieve")
        graph.add_conditional_edges("retrieve", self.should_continue, ["tool_node", END])
        graph.add_edge("tool_node", "retrieve")
        return graph
