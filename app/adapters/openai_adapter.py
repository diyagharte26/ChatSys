from typing import Dict, Any, List
from .base_adapter import BaseLLMAdapter

class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI API adapter"""
    
    BASE_URL = "https://api.openai.com/v1"
    
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call OpenAI API"""
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = self._prepare_payload(messages)
        response = await self._make_request(url, payload)
        
        # OpenAI returns in the correct format already
        return response
    
    def get_langchain_llm(self):
        """Get LangChain LLM for OpenAI"""
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                openai_api_key=self.api_key,
                model_name=self.model,
                temperature=self.temperature
            )
        except ImportError:
            return None
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenAI uses standard format"""
        return messages
    
    def _prepare_headers(self) -> Dict[str, str]:
        """OpenAI-specific headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

# Default models for OpenAI
OPENAI_MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k"
]