from typing import Dict, Any
from .groq_adapter import GroqAdapter
from .openai_adapter import OpenAIAdapter
from .gemini_adapter import GeminiAdapter
from .mistral_adapter import MistralAdapter
from .openrouter_adapter import OpenRouterAdapter
from .anthropic_adapter import AnthropicAdapter

# Registry of available adapters
ADAPTER_REGISTRY = {
    "groq": GroqAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "mistral": MistralAdapter,
    "openrouter": OpenRouterAdapter,
    "anthropic": AnthropicAdapter
}

def get_adapter(provider: str):
    """Get adapter instance for provider"""
    if provider not in ADAPTER_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider}")
    
    adapter_class = ADAPTER_REGISTRY[provider]
    return adapter_class()

def list_providers() -> list:
    """Get list of supported providers"""
    return list(ADAPTER_REGISTRY.keys())

__all__ = [
    "get_adapter",
    "list_providers",
    "GroqAdapter",
    "OpenAIAdapter", 
    "GeminiAdapter",
    "MistralAdapter",
    "OpenRouterAdapter",
    "AnthropicAdapter"
]