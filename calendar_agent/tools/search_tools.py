from typing import Dict, Any
import logging

from langchain_core.tools import tool

from calendar_agent.weaviate_db import find_related_appointments, find_similar_conversations

logger = logging.getLogger(__name__)

@tool
def find_similar_appointments(customer_name: str, search_description: str) -> Dict[str, Any]:
    """Find past appointments similar to a description for a specific customer.
    
    Use this to check if a new appointment might be related to previous work.
    
    Args:
        customer_name: Customer's full name
        search_description: Description of what you're looking for (can include service type, location, issue, etc.)
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - customer_name (str): The customer name searched
        - appointments (list): List of up to 5 similar appointment dictionaries with fields: appointment_id, title, address, description, start_time
        - count (int): Number of appointments found
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: find_similar_appointments(customer='{customer_name}')")
    try:
        # Use the search description as the query
        related = find_related_appointments(customer_name, search_description, search_description, "", limit=5)
        
        return {
            "success": True,
            "customer_name": customer_name,
            "appointments": related if related else [],
            "count": len(related) if related else 0
        }
    except Exception as e:
        logger.error(f"❌ Error in find_similar_appointments: {e}")
        return {"success": False, "error": f"Failed to find similar appointments - {str(e)}"}

@tool
def search_conversation_history(search_query: str) -> Dict[str, Any]:
    """Search through past conversation history to find relevant discussions.
    
    Use this when the user asks about previous conversations or when you need context from past discussions.
    
    Args:
        search_query: What to search for in past conversations (topics, questions, appointments discussed, etc.)
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - conversations (list): List of up to 5 relevant conversation dictionaries with fields: user_message, assistant_response, timestamp
        - count (int): Number of conversations found
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: search_conversation_history(query='{search_query}')")
    try:
        similar = find_similar_conversations(search_query, limit=5)
        
        return {
            "success": True,
            "conversations": similar if similar else [],
            "count": len(similar) if similar else 0
        }
    except Exception as e:
        logger.error(f"❌ Error in search_conversation_history: {e}")
        return {"success": False, "error": f"Failed to search conversation history - {str(e)}"}
