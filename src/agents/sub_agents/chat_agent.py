from typing import Dict, Any

from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, END

from src.core.base_agent import BaseAgent
from src.core.state import RAGAgentState, InputState, OutputState
from src.utils.loggers import logger

_SYSTEM_PROMPT = """You are a friendly, empathetic conversational assistant.

Your role is to respond naturally and helpfully to the user's message.

Tone guidelines based on detected user emotion:
- frustrated / concerned / anxious → calm, reassuring, solution-oriented
- happy                            → warm, enthusiastic, positive
- neutral                          → professional, clear, concise

Rules:
- Keep answers focused and conversational.
- Do not invent facts you do not know — admit uncertainty when needed.
- For greetings or small talk, engage naturally without over-explaining.
"""


class ChatAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="Chat Agent", system_prompt=_SYSTEM_PROMPT, **kwargs)

    def chat_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emotion = state.get("emotion") or "neutral"
        system_content = (
            self.system_prompt
            + f"\n\n[Detected user emotion: '{emotion}'. Adjust tone accordingly.]"
        )
        messages = [SystemMessage(content=system_content)]
        messages += trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=10_000,
            start_on="human",
            end_on=("human", "tool"),
        )
        response = self.model.invoke(messages)
        logger.info("ChatAgent.chat_node: emotion=%s", emotion)
        return {"messages": [response]}

    def build_graph(self):
        graph = StateGraph(RAGAgentState, input=InputState, output=OutputState)
        graph.add_node("chat", self.chat_node)
        graph.add_edge(START, "chat")
        graph.add_edge("chat", END)
        return graph
