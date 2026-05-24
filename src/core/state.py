from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from enum import Enum


class BaseState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    error: Optional[str]
    data: Optional[List]


class InputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class OutputState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    response: str
    metadata: Optional[Dict[str, Any]]
    dpp_data: Optional[List]
    sto_data: Optional[List]
    edit_data: Optional[List] = Field([], description="editable data for context")


class MainAWSAgentState(BaseState):
    metadata: Optional[Dict[str, Any]]


class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent response")
    thread_id: str = Field(..., description="Thread ID for conversation continuity")
    data: Optional[List] = Field([], description="data for context")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation continuity")
    user_id: str = Field(None, description="user Id")
    run_tool_simulation: Optional[bool] = Field(False, description="Tool testing for simulation")
    aws_context: Optional[str] = Field(None, description="aws context for the tool simulation")


class GetAgentState(BaseModel):
    thread_id: str = Field(None, description="Thread ID for conversation continuity")


class AWSTag(BaseModel):
    Key: str = Field(..., description="Tag key (required)")
    Value: Optional[str] = Field(None, description="Tag value (optional, can be None)")


class AWSConfigIntent(BaseModel):
    tags: Optional[List[AWSTag]] = Field(default=None, description="AWS tags as a list of key/value pairs")


class UserIntent(str, Enum):
    NEEDS_RAG = "needs_rag"
    GENERAL_CHAT = "general_chat"


class UserEmotion(str, Enum):
    FRUSTRATED = "frustrated"
    HAPPY = "happy"
    NEUTRAL = "neutral"
    CONCERNED = "concerned"
    ANXIOUS = "anxious"


class IntentEmotionAnalysis(BaseModel):
    intent: UserIntent = Field(..., description="User's intent classification")
    emotion: UserEmotion = Field(..., description="User's detected emotion")
    reasoning: str = Field(..., description="One-sentence reasoning for this classification")


class RAGAgentState(BaseState):
    intent: Optional[str]
    emotion: Optional[str]
    metadata: Optional[Dict[str, Any]]
