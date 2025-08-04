from typing import Dict, Any, List
from .base_adapter import BaseLLMAdapter

class GroqAdapter(BaseLLMAdapter):
    """Groq API adapter"""
    
    BASE_URL = "https://api.groq.com/openai/v1"
    
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Groq API"""
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = self._prepare_payload(messages)
        response = await self._make_request(url, payload)
        
        # Groq already returns OpenAI-compatible format
        return response
    
    def get_langchain_llm(self):
        """Get LangChain LLM for Groq"""
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model,
                temperature=self.temperature
            )
        except ImportError:
            # Fallback to generic LLM
            from langchain_community.llms import LlamaCpp
            return None
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groq uses standard OpenAI format"""
        return messages
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Groq-specific headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

# Default models for Groq
GROQ_MODELS = [
    "llama2-70b-4096",
    "mixtral-8x7b-32768", 
    "gemma-7b-it",
    "llama3-8b-8192",
    "llama3-70b-8192"
]