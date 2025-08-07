import uuid
import time
from typing import Dict, Any, List
from .models import ChatRequest, ChatResponse, ChatChoice, ChatMessage, ChatUsage
from .llm_config import get_user_llm_config
from .agent import get_agent
from .adapters import get_adapter
from .response_utils import convert_to_openai_format, count_tokens
from .database import DatabaseManager

async def process_chat_request(user_id: int, request: ChatRequest) -> ChatResponse:
    """Process chat request and return OpenAI-compatible response"""
    
    # Get user's LLM configuration
    llm_config = await get_user_llm_config(user_id)
    if not llm_config:
        raise ValueError("No LLM configuration found. Please configure your LLM provider first.")
    
    # Override model and temperature if provided in request
    if request.model:
        llm_config["model"] = request.model
    if request.temperature is not None:
        llm_config["temperature"] = request.temperature
    
    # Validate messages before accessing
    if not request.messages or len(request.messages) == 0:
        raise ValueError("The messages list is empty.")

    # Convert messages
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Generate session ID
    session_id = uuid.uuid4().hex

    # Get adapter and configure
    adapter = get_adapter(llm_config["provider"])
    adapter.configure(
        api_key=llm_config["api_key"],
        model=llm_config["model"],
        temperature=llm_config["temperature"]
    )

    # Get agent
    agent = get_agent(adapter)

    # Process with agent
    agent_response = await agent.process_messages(messages)

    # Convert to OpenAI format
    openai_response = convert_to_openai_format(
        session_id=session_id,
        model=llm_config["model"],
        agent_response=agent_response,
        request_messages=messages
    )

    # Save session
    await DatabaseManager.save_session(
        session_id=session_id,
        user_id=user_id,
        messages=messages,
        response=openai_response.dict(),
        tools_used=agent_response.tools_used
    )

    return openai_response


def create_error_response(error_message: str, session_id: str = None) -> ChatResponse:
    """Create error response in OpenAI format"""
    if not session_id:
        session_id = uuid.uuid4().hex
    
    return ChatResponse(
        id=session_id,
        object="chat.completion",
        created=int(time.time()),
        model="error",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=f"Error: {error_message}"
                ),
                finish_reason="error"
            )
        ],
        usage=ChatUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )

async def validate_chat_request(request: ChatRequest) -> bool:
    """Validate chat request"""
    if not request.messages:
        return False
    
    if len(request.messages) == 0:
        return False
    
    # Check message roles
    for msg in request.messages:
        if msg.role not in ["system", "user", "assistant"]:
            return False
    
    return True

def format_messages_for_llm(messages: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    """Format messages for specific LLM provider"""
    formatted_messages = []
    
    for msg in messages:
        formatted_msg = {
            "role": msg["role"],
            "content": msg["content"]
        }
        
        # Add provider-specific formatting if needed
        if provider == "gemini":
            # Gemini uses "user" and "model" instead of "user" and "assistant"
            if formatted_msg["role"] == "assistant":
                formatted_msg["role"] = "model"
        elif provider == "anthropic":
            # Anthropic has specific formatting requirements
            if formatted_msg["role"] == "system":
                # Anthropic handles system messages differently
                formatted_msg["role"] = "user"
                formatted_msg["content"] = f"System: {formatted_msg['content']}"
        
        formatted_messages.append(formatted_msg)
    
    return formatted_messages

def extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    """Extract the last user message for agent processing"""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""

def build_context_for_agent(messages: List[Dict[str, Any]]) -> str:
    """Build context string from conversation history"""
    context_parts = []
    
    for msg in messages[:-1]:  # Exclude the last message
        role = msg["role"].title()
        content = msg["content"]
        context_parts.append(f"{role}: {content}")
    
    return "\n\n".join(context_parts)