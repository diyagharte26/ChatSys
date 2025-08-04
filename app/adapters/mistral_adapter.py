from typing import Dict, Any, List
from .base_adapter import BaseLLMAdapter

class MistralAdapter(BaseLLMAdapter):
    """Mistral API adapter"""
    
    BASE_URL = "https://api.mistral.ai/v1"
    
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Mistral API"""
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = self._prepare_payload(messages)
        response = await self._make_request(url, payload)
        
        # Mistral returns OpenAI-compatible format
        return response
    
    def get_langchain_llm(self):
        """Get LangChain LLM for Mistral"""
        try:
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(
                mistral_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature
            )
        except ImportError:
            return None
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mistral uses standard OpenAI format"""
        return messages
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Mistral-specific headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

# Default models for Mistral
MISTRAL_MODELS = [
    "mistral-large-latest",
    "mistral-medium-latest", 
    "mistral-small-latest",
    "mistral-tiny",
    "open-mistral-7b",
    "open-mixtral-8x7b"
]