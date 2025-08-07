from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

from .database import get_db, init_db, DatabaseManager
from .models import UserCreate, UserLogin, LLMConfigCreate, ChatRequest, ChatResponse
from .auth import create_user, authenticate_user, get_user_by_api_key
from .llm_config import save_llm_config, get_user_llm_config
from .chat import process_chat_request


app = FastAPI(
    title="KubeSage API",
    description="AI-powered Kubernetes automation tool using ChatOps",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Authenticate user by API key"""
    api_key = credentials.credentials
    user = await get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user

@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    await init_db()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "KubeSage API is running", "version": "1.0.0"}

@app.post("/register")
async def register(user_data: UserCreate):
    """Register new user and return API key"""
    try:
        api_key = await create_user(user_data.username, user_data.password)
        return {"api_key": f"Bearer {api_key}", "message": "User created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(login_data: UserLogin):
    """Login user and return existing API key"""
    api_key = await authenticate_user(login_data.username, login_data.password)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"api_key": f"Bearer {api_key}", "message": "Login successful"}

@app.post("/config")
async def configure_llm(
    config_data: LLMConfigCreate,
    current_user = Depends(get_current_user)
):
    """Save LLM configuration for user"""
    try:
        await save_llm_config(current_user["id"], config_data)
        return {"message": "LLM configuration saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    current_user = Depends(get_current_user)
):
    """Main chat endpoint - OpenAI compatible"""
    try:
        response = await process_chat_request(current_user["id"], request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions(current_user = Depends(get_current_user)):
    """Get all session IDs for user"""
    sessions = await DatabaseManager.get_user_sessions(current_user["id"])
    return {"sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    current_user = Depends(get_current_user)
):
    """Get full conversation history for a session"""
    session = await DatabaseManager.get_session_by_id(current_user["id"], session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)