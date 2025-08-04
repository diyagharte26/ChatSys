import time
import uuid
from typing import Dict, Any, List
from .models import ChatResponse, ChatChoice, ChatMessage, ChatUsage, AgentResponse

def convert_to_openai_format(
    session_id: str,
    model: str,
    agent_response: AgentResponse,
    request_messages: List[Dict[str, Any]]
) -> ChatResponse:
    """Convert agent response to OpenAI chat completion format"""
    
    # Count tokens (simplified estimation)
    prompt_tokens = sum(len(msg["content"].split()) for msg in request_messages)
    completion_tokens = len(agent_response.content.split())
    
    return ChatResponse(
        id=session_id,
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=agent_response.content
                ),
                finish_reason="stop"
            )
        ],
        usage=ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

def count_tokens(text: str) -> int:
    """Simple token counting (words as approximation)"""
    return len(text.split())

def create_streaming_response(content: str, session_id: str, model: str):
    """Create streaming response chunks (for future streaming support)"""
    chunks = []
    words = content.split()
    
    for i, word in enumerate(words):
        chunk = {
            "id": session_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": word + " " if i < len(words) - 1 else word
                    },
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }
            ]
        }
        chunks.append(chunk)
    
    return chunks

def format_error_response(error_message: str, session_id: str = None) -> ChatResponse:
    """Format error as OpenAI-compatible response"""
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

def extract_content_from_llm_response(response: Dict[str, Any]) -> str:
    """Extract content from various LLM response formats"""
    
    # OpenAI format
    if "choices" in response:
        choices = response["choices"]
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            return message.get("content", "")
    
    # Direct content
    if "content" in response:
        return response["content"]
    
    # Anthropic format
    if "completion" in response:
        return response["completion"]
    
    # Gemini format
    if "candidates" in response:
        candidates = response["candidates"]
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                return parts[0].get("text", "")
    
    return str(response)

def calculate_usage_stats(request_messages: List[Dict], response_content: str) -> ChatUsage:
    """Calculate token usage statistics"""
    
    # Simple estimation based on word count
    prompt_text = " ".join([msg["content"] for msg in request_messages])
    prompt_tokens = max(1, len(prompt_text.split()) * 1.3)  # Rough token estimation
    completion_tokens = max(1, len(response_content.split()) * 1.3)
    
    return ChatUsage(
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
        total_tokens=int(prompt_tokens + completion_tokens)
    )

def normalize_message_format(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize message format across different providers"""
    normalized = []
    
    for msg in messages:
        normalized_msg = {
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        }
        
        # Add name if present
        if "name" in msg:
            normalized_msg["name"] = msg["name"]
        
        normalized.append(normalized_msg)
    
    return normalized

def add_system_context(messages: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
    """Add system context to message list"""
    if not context:
        return messages
    
    # Check if first message is already system message
    if messages and messages[0]["role"] == "system":
        # Append to existing system message
        messages[0]["content"] = f"{messages[0]['content']}\n\n{context}"
        return messages
    else:
        # Add new system message at the beginning
        system_msg = {"role": "system", "content": context}
        return [system_msg] + messages

def format_tool_usage_summary(tools_used: List[str]) -> str:
    """Format summary of tools used in the response"""
    if not tools_used:
        return ""
    
    if len(tools_used) == 1:
        return f"\n\n*Used tool: {tools_used[0]}*"
    else:
        tool_list = ", ".join(tools_used)
        return f"\n\n*Used tools: {tool_list}*"