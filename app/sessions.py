from typing import List, Dict, Any, Optional
from .database import DatabaseManager

async def get_user_sessions(user_id: int) -> List[Dict[str, Any]]:
    """Get all sessions for a user"""
    return await DatabaseManager.get_user_sessions(user_id)

async def get_session_detail(user_id: int, session_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed session information"""
    return await DatabaseManager.get_session_by_id(user_id, session_id)

async def create_session_summary(session: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of a session"""
    if not session:
        return {}
    
    messages = session.get("messages", [])
    tools_used = session.get("tools_used", [])
    
    # Extract first user message as title
    title = "Untitled Session"
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            title = content[:50] + "..." if len(content) > 50 else content
            break
    
    return {
        "session_id": session["session_id"],
        "title": title,
        "timestamp": session["timestamp"],
        "message_count": len(messages),
        "tools_used_count": len(tools_used),
        "tools_used": tools_used
    }

async def get_recent_sessions(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent sessions with summaries"""
    sessions = await DatabaseManager.get_user_sessions(user_id)
    
    # Limit results
    recent_sessions = sessions[:limit]
    
    # Get detailed info for each session to create summaries
    summaries = []
    for session_info in recent_sessions:
        session_detail = await get_session_detail(user_id, session_info["session_id"])
        if session_detail:
            summary = await create_session_summary(session_detail)
            summaries.append(summary)
    
    return summaries

async def delete_session(user_id: int, session_id: str) -> bool:
    """Delete a session"""
    try:
        query = "DELETE FROM sessions WHERE user_id = ? AND session_id = ?"
        result = await DatabaseManager.execute_query(query, (user_id, session_id))
        return result is not None
    except Exception:
        return False

async def search_sessions(
    user_id: int, 
    query: str, 
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Search sessions by content"""
    try:
        # Simple text search in messages
        search_query = """
            SELECT session_id, timestamp, messages 
            FROM sessions 
            WHERE user_id = ? AND messages LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        search_pattern = f"%{query}%"
        results = await DatabaseManager.execute_query(
            search_query, 
            (user_id, search_pattern, limit), 
            fetch_all=True
        )
        
        return results or []
    
    except Exception:
        return []

async def get_session_statistics(user_id: int) -> Dict[str, Any]:
    """Get session statistics for user"""
    try:
        # Total sessions
        total_query = "SELECT COUNT(*) as total FROM sessions WHERE user_id = ?"
        total_result = await DatabaseManager.execute_query(total_query, (user_id,), fetch_one=True)
        total_sessions = total_result["total"] if total_result else 0
        
        # Total messages (approximate)
        all_sessions = await DatabaseManager.get_user_sessions(user_id)
        total_messages = 0
        total_tools_used = 0
        
        for session_info in all_sessions:
            session_detail = await get_session_detail(user_id, session_info["session_id"])
            if session_detail:
                total_messages += len(session_detail.get("messages", []))
                total_tools_used += len(session_detail.get("tools_used", []))
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_tools_used": total_tools_used,
            "average_messages_per_session": total_messages / max(1, total_sessions)
        }
    
    except Exception:
        return {
            "total_sessions": 0,
            "total_messages": 0,
            "total_tools_used": 0,
            "average_messages_per_session": 0
        }

async def export_session(user_id: int, session_id: str) -> Optional[Dict[str, Any]]:
    """Export session data in a portable format"""
    session = await get_session_detail(user_id, session_id)
    
    if not session:
        return None
    
    return {
        "session_id": session["session_id"],
        "timestamp": session["timestamp"],
        "messages": session["messages"],
        "response": session["response"],
        "tools_used": session["tools_used"],
        "export_version": "1.0"
    }

def format_session_for_display(session: Dict[str, Any]) -> Dict[str, Any]:
    """Format session for frontend display"""
    if not session:
        return {}
    
    # Format messages for better display
    formatted_messages = []
    for msg in session.get("messages", []):
        formatted_msg = {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": session["timestamp"]  # Add session timestamp to each message
        }
        formatted_messages.append(formatted_msg)
    
    return {
        "session_id": session["session_id"],
        "timestamp": session["timestamp"],
        "messages": formatted_messages,
        "tools_used": session.get("tools_used", []),
        "summary": {
            "message_count": len(formatted_messages),
            "tool_count": len(session.get("tools_used", []))
        }
    }