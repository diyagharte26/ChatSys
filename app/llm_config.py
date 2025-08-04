from typing import Optional, Dict, Any
from .database import DatabaseManager
from .models import LLMConfigCreate

# Supported LLM providers
SUPPORTED_PROVIDERS = {
    "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
    "groq": ["llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"],
    "gemini": ["gemini-pro", "gemini-pro-vision"],
    "mistral": ["mistral-7b-instruct", "mistral-medium", "mistral-large"],
    "openrouter": ["meta-llama/llama-2-70b-chat", "anthropic/claude-2"],
    "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
}

def validate_provider_model(provider: str, model: str) -> bool:
    """Validate that provider supports the model"""
    if provider not in SUPPORTED_PROVIDERS:
        return False
    
    # For now, accept any model for flexibility
    # In production, you might want stricter validation
    return True

async def save_llm_config(user_id: int, config: LLMConfigCreate) -> None:
    """Save LLM configuration for user"""
    
    # Validate provider and model
    if not validate_provider_model(config.provider, config.model):
        raise ValueError(f"Unsupported provider/model combination: {config.provider}/{config.model}")
    
    # Validate temperature
    if not 0.0 <= config.temperature <= 2.0:
        raise ValueError("Temperature must be between 0.0 and 2.0")
    
    # Save to database
    await DatabaseManager.save_llm_config(
        user_id=user_id,
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature
    )

async def get_user_llm_config(user_id: int) -> Optional[Dict[str, Any]]:
    """Get LLM configuration for user"""
    return await DatabaseManager.get_llm_config(user_id)

def get_supported_providers() -> Dict[str, list]:
    """Get list of supported providers and their models"""
    return SUPPORTED_PROVIDERS

async def update_llm_config(user_id: int, config: LLMConfigCreate) -> None:
    """Update existing LLM configuration"""
    # This uses the same save function as it does INSERT OR REPLACE
    await save_llm_config(user_id, config)

async def delete_llm_config(user_id: int) -> None:
    """Delete LLM configuration for user"""
    query = "DELETE FROM llm_configs WHERE user_id = ?"
    await DatabaseManager.execute_query(query, (user_id,))

def get_default_config() -> Dict[str, Any]:
    """Get default LLM configuration"""
    return {
        "provider": "groq",
        "model": "mixtral-8x7b-32768",
        "temperature": 0.7
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate LLM configuration structure"""
    required_fields = ["provider", "model", "api_key"]
    
    for field in required_fields:
        if field not in config:
            return False
    
    if "temperature" in config:
        if not isinstance(config["temperature"], (int, float)):
            return False
        if not 0.0 <= config["temperature"] <= 2.0:
            return False
    
    return True