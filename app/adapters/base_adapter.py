from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import httpx

class BaseLLMAdapter(ABC):
    """Base class for all LLM adapters"""
    
    def __init__(self):
        self.api_key: Optional[str] = None
        self.model: Optional[str] = None
        self.temperature: float = 0.7
        self.client: Optional[httpx.AsyncClient] = None
    
    def configure(self, api_key: str, model: str, temperature: float = 0.7):
        """Configure the adapter with credentials and settings"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = httpx.AsyncClient()
    
    @abstractmethod
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make raw API call and return response in OpenAI-compatible format"""
        pass
    
    @abstractmethod
    def get_langchain_llm(self):
        """Get LangChain LLM instance for agent use"""
        pass
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for this provider (override if needed)"""
        return messages
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare common headers for API requests"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _prepare_payload(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Prepare common payload structure"""
        return {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": self.temperature,
            **kwargs
        }
    
    async def _make_request(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to LLM API"""
        if not self.client:
            raise RuntimeError("Adapter not configured. Call configure() first.")
        
        headers = self._prepare_headers()
        
        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise Exception(f"API request failed: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def _convert_to_openai_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert provider response to OpenAI format (override per provider)"""
        # Default implementation - should be overridden
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(response)
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

class MockAdapter(BaseLLMAdapter):
    """Mock adapter for testing"""
    
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock implementation"""
        last_message = messages[-1]["content"] if messages else "Hello"
        
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Mock response to: {last_message}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
    
    def get_langchain_llm(self):
        """Return None for mock"""
        return None