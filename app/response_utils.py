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
    """Extract content from various LLM response formats in a safe manner."""
    
    try:
        # OpenAI / Groq / Mistral / OpenRouter format
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return content

        # Anthropic format (new API, e.g., Claude 3)
        anthropic_content_list = response.get("content", [])
        if isinstance(anthropic_content_list, list) and len(anthropic_content_list) > 0:
            text_content = []
            for block in anthropic_content_list:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content.append(block.get("text", ""))
            if text_content:
                return "".join(text_content)

        # Anthropic format (legacy API)
        completion = response.get("completion")
        if completion:
            return completion

        # Gemini format
        candidates = response.get("candidates", [])
        if candidates and len(candidates) > 0:
            candidate = candidates[0]
            if isinstance(candidate, dict):
                content_data = candidate.get("content", {})
                if isinstance(content_data, dict):
                    parts = content_data.get("parts", [])
                    if parts and len(parts) > 0:
                        part = parts[0]
                        if isinstance(part, dict):
                            text = part.get("text")
                            if text:
                                return text
        
        # Direct content key (as a fallback if other structures fail)
        direct_content = response.get("content")
        if isinstance(direct_content, str):
            return direct_content

    except Exception:
        # A final catch-all, though the above logic should prevent most errors.
        pass

    # If no content could be extracted, return the raw response for debugging.
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
