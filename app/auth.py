import hashlib
import uuid
from typing import Optional, Dict, Any
from .database import DatabaseManager

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed_password

def generate_api_key() -> str:
    """Generate unique API key"""
    return uuid.uuid4().hex

async def create_user(username: str, password: str) -> str:
    """Create new user and return API key"""
    # Check if user already exists
    existing_user = await DatabaseManager.get_user_by_username(username)
    if existing_user:
        raise ValueError("Username already exists")
    
    # Hash password and generate API key
    password_hash = hash_password(password)
    api_key = generate_api_key()
    
    # Create user in database
    user_id = await DatabaseManager.create_user(username, password_hash, api_key)
    
    if not user_id:
        raise ValueError("Failed to create user")
    
    return api_key

async def authenticate_user(username: str, password: str) -> Optional[str]:
    """Authenticate user and return API key"""
    user = await DatabaseManager.get_user_by_username(username)
    
    if not user:
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    return user["api_key"]

async def get_user_by_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Get user by API key"""
    return await DatabaseManager.get_user_by_api_key(api_key)

async def validate_api_key(api_key: str) -> bool:
    """Validate API key exists"""
    user = await get_user_by_api_key(api_key)
    return user is not None

async def get_user_id_from_api_key(api_key: str) -> Optional[int]:
    """Get user ID from API key"""
    user = await get_user_by_api_key(api_key)
    return user["id"] if user else None