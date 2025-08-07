from typing import Dict, Any, List
from .base_adapter import BaseLLMAdapter

class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini API adapter"""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    async def call_raw(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Gemini API"""
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        
        # Gemini has different format
        payload = self._prepare_gemini_payload(messages)
        response = await self._make_request(url, payload)
        
        # Convert Gemini response to OpenAI format
        return self._convert_gemini_to_openai(response)
    
    def get_langchain_llm(self):
        """Get LangChain LLM for Gemini"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=self.model,
                temperature=self.temperature
            )
        except ImportError:
            return None
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert to Gemini format"""
        gemini_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Gemini uses "user" and "model" instead of "user" and "assistant"
            if role == "assistant":
                role = "model"
            elif role == "system":
                # Gemini doesn't have system role, prepend to first user message
                role = "user"
                content = f"System: {content}"
            
            gemini_messages.append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        return gemini_messages
    
    def _prepare_gemini_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare Gemini-specific payload"""
        formatted_messages = self.format_messages(messages)
        
        return {
            "contents": formatted_messages,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048,
            }
        }
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Gemini-specific headers"""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
    
    def _convert_gemini_to_openai(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Gemini response to OpenAI format"""
        try:
            # Extract content from Gemini response
            candidates = response.get("candidates", [])
            if not candidates or len(candidates) == 0:
                content = "No response generated"
            else:
                parts = candidates[0].get("content", {}).get("parts", [])
                content = parts[0].get("text", "") if parts and len(parts) > 0 else "No content"
            
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.get("usageMetadata", {}).get("promptTokenCount", 0),
                    "completion_tokens": response.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                    "total_tokens": response.get("usageMetadata", {}).get("totalTokenCount", 0)
                }
            }
        except Exception as e:
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Error parsing Gemini response: {str(e)}"
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

# Default models for Gemini
GEMINI_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-pro",
    "gemini-pro-vision"
]