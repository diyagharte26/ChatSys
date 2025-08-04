from typing import Dict, Any, List
from .base_adapter import BaseLLMAdapter

class OpenRouterAdapter(BaseLLMAdapter):
    """OpenRouter API adapter"""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call OpenRouter API"""
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = self._prepare_payload(messages)
        response = await self._make_request(url, payload)
        
        # OpenRouter returns OpenAI-compatible format
        return response
    
    def get_langchain_llm(self):
        """Get LangChain LLM for OpenRouter"""
        try:
            from langchain_openai import ChatOpenAI
            # Use OpenAI client with OpenRouter endpoint
            return ChatOpenAI(
                openai_api_key=self.api_key,
                openai_api_base=self.BASE_URL,
                model_name=self.model,
                temperature=self.temperature
            )
        except ImportError:
            return None
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenRouter uses standard OpenAI format"""
        return messages
    
    def _prepare_headers(self) -> Dict[str, str]:
        """OpenRouter-specific headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://kubesage.ai",  # Optional: your app name
            "X-Title": "KubeSage"  # Optional: your app name
        }

# Popular models available on OpenRouter
OPENROUTER_MODELS = [
    "meta-llama/llama-2-70b-chat",
    "anthropic/claude-2",
    "anthropic/claude-instant-v1",
    "openai/gpt-4",
    "openai/gpt-3.5-turbo",
    "google/palm-2-chat-bison",
    "mistralai/mistral-7b-instruct"
]