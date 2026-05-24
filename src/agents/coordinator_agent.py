from typing import Dict, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, END

from src.core.base_agent import BaseAgent
from src.core.state import (
    RAGAgentState,
    InputState,
    OutputState,
    UserIntent,
    UserEmotion,
    IntentEmotionAnalysis,
)
from src.agents.sub_agents.chat_agent import ChatAgent
from src.agents.sub_agents.rag_agent import RAGAgent
from src.utils.loggers import logger

_COORDINATOR_PROMPT = """You are an intent and emotion classifier for a conversational AI system.

Given the user's message, determine:

1. INTENT — choose exactly one:
   - needs_rag    : The user asks a factual, knowledge-based, or domain-specific question
                    that requires retrieving information (explanations, how-tos, troubleshooting,
                    definitions, document content, technical details).
   - general_chat : Greetings, small talk, gratitude, casual conversation, off-topic messages.

2. EMOTION — choose exactly one:
   - frustrated : expresses frustration, anger, or disappointment
   - anxious    : stressed, urgent, time-sensitive
   - concerned  : worried, uncertain, seeking reassurance
   - happy      : positive tone, excitement, satisfaction
   - neutral    : no strong emotional signal

Provide a one-sentence reasoning to justify your classification.
"""


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Coordinator Agent",
            system_prompt=_COORDINATOR_PROMPT,
            **kwargs,
        )
        self.chat_agent = ChatAgent()
        self.rag_agent = RAGAgent()
        self._classifier = self.model.with_structured_output(IntentEmotionAnalysis)

    def analyze_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        recent = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=4_000,
            start_on="human",
            end_on="human",
        )
        try:
            analysis: IntentEmotionAnalysis = self._classifier.invoke(
                [SystemMessage(content=self.system_prompt), *recent]
            )
            logger.info(
                "Coordinator.analyze: intent=%s emotion=%s reasoning='%s'",
                analysis.intent.value,
                analysis.emotion.value,
                analysis.reasoning,
            )
            return {
                "intent": analysis.intent.value,
                "emotion": analysis.emotion.value,
            }
        except Exception as e:
            logger.warning("Intent/emotion analysis failed (%s) — defaulting to needs_rag/neutral", e)
            return {
                "intent": UserIntent.NEEDS_RAG.value,
                "emotion": UserEmotion.NEUTRAL.value,
            }

    def _route_intent(self, state: Dict[str, Any]) -> Literal["rag_retrieve", "chat_respond"]:
        intent = state.get("intent", UserIntent.NEEDS_RAG.value)
        destination = "rag_retrieve" if intent == UserIntent.NEEDS_RAG.value else "chat_respond"
        logger.info("Coordinator routing → %s", destination)
        return destination

    def _rag_continue(self, state: Dict[str, Any]) -> Literal["rag_tool", "__end__"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "rag_tool"
        return END

    def build_graph(self):
        graph = StateGraph(RAGAgentState, input=InputState, output=OutputState)

        graph.add_node("analyze", self.analyze_node)
        graph.add_node("rag_retrieve", self.rag_agent.retrieve_node)
        graph.add_node("rag_tool", self.rag_agent.tool_node)
        graph.add_node("chat_respond", self.chat_agent.chat_node)

        graph.add_edge(START, "analyze")
        graph.add_conditional_edges(
            "analyze",
            self._route_intent,
            {"rag_retrieve": "rag_retrieve", "chat_respond": "chat_respond"},
        )
        graph.add_conditional_edges(
            "rag_retrieve",
            self._rag_continue,
            {"rag_tool": "rag_tool", END: END},
        )
        graph.add_edge("rag_tool", "rag_retrieve")
        graph.add_edge("chat_respond", END)

        return graph


class RAGAgentManager:
    def __init__(self):
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        logger.info("RAGAgentManager: initialising coordinator")
        self.coordinator = CoordinatorAgent()
        self.compiled = self.coordinator.compile()
        self._initialized = True
        logger.info("RAGAgentManager: ready")

    def process_message(self, message: str, thread_id: str, **kwargs) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        logger.info(
            "RAGAgentManager.process_message: thread_id=%s user_id=%s",
            thread_id,
            kwargs.get("user_id"),
        )

        result = self.compiled.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=message,
                        response_metadata={"user_id": kwargs.get("user_id")},
                    )
                ]
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        logger.info("RAGAgentManager: response ready for thread_id=%s", thread_id)
        return {"result": result, "thread_id": thread_id}
