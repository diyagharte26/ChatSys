from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

# User models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: int
    username: str
    api_key: str
    created_at: datetime

# LLM Configuration models
class LLMConfigCreate(BaseModel):
    provider: str = Field(..., description="LLM provider (groq, openai, gemini, mistral, openrouter)")
    model: str = Field(..., description="Model name")
    api_key: str = Field(..., description="API key for the provider")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class LLMConfig(BaseModel):
    id: int
    user_id: int
    provider: str
    model: str
    api_key: str
    temperature: float
    created_at: datetime

# Chat models (OpenAI compatible)
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage

# Session models
class SessionSummary(BaseModel):
    session_id: str
    timestamp: datetime

class SessionDetail(BaseModel):
    id: int
    session_id: str
    user_id: int
    messages: List[Dict[str, Any]]
    response: Dict[str, Any]
    tools_used: List[str]
    timestamp: datetime

# Tool execution models
class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    
# Agent response models
class AgentResponse(BaseModel):
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tools_used: List[str] = []
    
# Database models (for internal use)
class DBUser(BaseModel):
    id: int
    username: str
    password_hash: str
    api_key: str
    created_at: str

class DBLLMConfig(BaseModel):
    id: int
    user_id: int
    provider: str
    model: str
    api_key: str
    temperature: float
    created_at: str

class DBSession(BaseModel):
    id: int
    session_id: str
    user_id: int
    messages: str  # JSON string
    response: str  # JSON string
    tools_used: str  # JSON string
    timestamp: str