from typing import Dict, Any, List
from .base_adapter import BaseLLMAdapter

class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude API adapter"""
    
    BASE_URL = "https://api.anthropic.com/v1"
    
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Anthropic API"""
        url = f"{self.BASE_URL}/messages"
        
        # Anthropic has different format
        payload = self._prepare_anthropic_payload(messages)
        response = await self._make_request(url, payload)
        
        # Convert Anthropic response to OpenAI format
        return self._convert_anthropic_to_openai(response)
    
    def get_langchain_llm(self):
        """Get LangChain LLM for Anthropic"""
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                anthropic_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature
            )
        except ImportError:
            return None
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert to Anthropic format"""
        anthropic_messages = []
        system_content = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Anthropic handles system messages separately
                system_content = content
            else:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
        
        return anthropic_messages, system_content
    
    def _prepare_anthropic_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare Anthropic-specific payload"""
        formatted_messages, system_content = self.format_messages(messages)
        
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "messages": formatted_messages
        }
        
        if system_content:
            payload["system"] = system_content
        
        return payload
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Anthropic-specific headers"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def _convert_anthropic_to_openai(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic response to OpenAI format"""
        try:
            # Extract content from Anthropic response
            content_blocks = response.get("content", [])
            content = ""
            
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")
            
            if not content:
                content = "No response generated"
            
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": response.get("stop_reason", "stop")
                    }
                ],
                "usage": {
                    "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                    "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0)
                }
            }
        except Exception as e:
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Error parsing Anthropic response: {str(e)}"
                        },
                        "finish_reason": "error"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

# Default models for Anthropic
ANTHROPIC_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2"
]