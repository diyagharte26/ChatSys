import sqlite3
import aiosqlite
import json
from typing import Dict, Any, List, Optional

DATABASE_URL = "kubesage.db"

async def init_db():
    """Initialize SQLite database with required tables"""
    async with aiosqlite.connect(DATABASE_URL) as db:
        # Users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                api_key TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # LLM configurations table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS llm_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                api_key TEXT NOT NULL,
                temperature REAL DEFAULT 0.7,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id)
            )
        """)
        
        # Sessions table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                messages TEXT NOT NULL,
                response TEXT NOT NULL,
                tools_used TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        await db.commit()

async def get_db():
    """Get database connection"""
    return aiosqlite.connect(DATABASE_URL)

class DatabaseManager:
    """Helper class for database operations"""
    
    @staticmethod
    async def execute_query(query: str, params: tuple = (), fetch_one: bool = False, fetch_all: bool = False):
        """Execute database query"""
        async with aiosqlite.connect(DATABASE_URL) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            
            if fetch_one:
                result = await cursor.fetchone()
                return dict(result) if result else None
            elif fetch_all:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                await db.commit()
                return cursor.lastrowid

    @staticmethod
    async def create_user(username: str, password_hash: str, api_key: str) -> int:
        """Create new user"""
        query = "INSERT INTO users (username, password_hash, api_key) VALUES (?, ?, ?)"
        return await DatabaseManager.execute_query(query, (username, password_hash, api_key))

    @staticmethod
    async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        query = "SELECT * FROM users WHERE username = ?"
        return await DatabaseManager.execute_query(query, (username,), fetch_one=True)

    @staticmethod
    async def get_user_by_api_key(api_key: str) -> Optional[Dict[str, Any]]:
        """Get user by API key"""
        query = "SELECT * FROM users WHERE api_key = ?"
        return await DatabaseManager.execute_query(query, (api_key,), fetch_one=True)

    @staticmethod
    async def save_llm_config(user_id: int, provider: str, model: str, api_key: str, temperature: float):
        """Save or update LLM configuration"""
        query = """
            INSERT OR REPLACE INTO llm_configs (user_id, provider, model, api_key, temperature)
            VALUES (?, ?, ?, ?, ?)
        """
        return await DatabaseManager.execute_query(query, (user_id, provider, model, api_key, temperature))

    @staticmethod
    async def get_llm_config(user_id: int) -> Optional[Dict[str, Any]]:
        """Get LLM configuration for user"""
        query = "SELECT * FROM llm_configs WHERE user_id = ?"
        return await DatabaseManager.execute_query(query, (user_id,), fetch_one=True)

    @staticmethod
    async def save_session(session_id: str, user_id: int, messages: List[Dict], response: Dict, tools_used: List[str]):
        """Save chat session"""
        query = """
            INSERT INTO sessions (session_id, user_id, messages, response, tools_used)
            VALUES (?, ?, ?, ?, ?)
        """
        return await DatabaseManager.execute_query(
            query,
            (
                session_id,
                user_id,
                json.dumps(messages),
                json.dumps(response),
                json.dumps(tools_used)
            )
        )

    @staticmethod
    async def get_user_sessions(user_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for user"""
        query = "SELECT session_id, timestamp FROM sessions WHERE user_id = ? ORDER BY timestamp DESC"
        return await DatabaseManager.execute_query(query, (user_id,), fetch_all=True)

    @staticmethod
    async def get_session_by_id(user_id: int, session_id: str) -> Optional[Dict[str, Any]]:
        """Get specific session"""
        query = "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?"
        result = await DatabaseManager.execute_query(query, (user_id, session_id), fetch_one=True)
        
        if result:
            # Parse JSON fields
            result["messages"] = json.loads(result["messages"])
            result["response"] = json.loads(result["response"])
            result["tools_used"] = json.loads(result["tools_used"]) if result["tools_used"] else []
        
        return result

