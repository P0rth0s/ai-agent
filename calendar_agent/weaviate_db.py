from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import logging
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from psycopg2.extras import RealDictCursor

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Weaviate client (for long-term memory)
_weaviate_client = None

def get_weaviate_client():
    """Get or create Weaviate client for vector storage"""
    global _weaviate_client
    if _weaviate_client is None:
        try:
            _weaviate_client = weaviate.connect_to_local(
                host=os.getenv("WEAVIATE_HOST", "localhost"),
                port=int(os.getenv("WEAVIATE_PORT", "8080"))
            )
            logger.info("✅ Connected to Weaviate")
            _initialize_weaviate_collections()
        except Exception as e:
            logger.error(f"❌ Failed to connect to Weaviate: {e}")
    return _weaviate_client

def close_weaviate_client():
    """Close the Weaviate client connection properly"""
    global _weaviate_client
    if _weaviate_client is not None:
        try:
            _weaviate_client.close()
            logger.info("✅ Weaviate connection closed")
            _weaviate_client = None
        except Exception as e:
            logger.error(f"❌ Error closing Weaviate connection: {e}")

def _initialize_weaviate_collections():
    """Initialize Weaviate collections for appointment storage"""
    try:
        client = _weaviate_client
        
        # Appointment History Collection for semantic search
        if not client.collections.exists("AppointmentHistory"):
            try:
                client.collections.create(
                    name="AppointmentHistory",
                    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                    properties=[
                        Property(name="appointment_id", data_type=DataType.INT),
                        Property(name="customer_name", data_type=DataType.TEXT),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="description", data_type=DataType.TEXT),
                        Property(name="address", data_type=DataType.TEXT),
                        Property(name="start_time", data_type=DataType.DATE),
                        Property(name="completion_date", data_type=DataType.DATE),
                        Property(name="combined_text", data_type=DataType.TEXT),  # For vectorization
                    ]
                )
                logger.info("✅ Created AppointmentHistory collection")
            except Exception as create_error:
                if "already exists" in str(create_error).lower():
                    logger.info("ℹ️ AppointmentHistory collection already exists")
                else:
                    raise
        
        # Chat History Collection for long-term conversation memory
        if not client.collections.exists("ChatHistory"):
            try:
                client.collections.create(
                    name="ChatHistory",
                    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                    properties=[
                        Property(name="session_id", data_type=DataType.TEXT),
                        Property(name="user_message", data_type=DataType.TEXT),
                        Property(name="assistant_response", data_type=DataType.TEXT),
                        Property(name="timestamp", data_type=DataType.DATE),
                        Property(name="conversation_context", data_type=DataType.TEXT),  # Combined text for vectorization
                    ]
                )
                logger.info("✅ Created ChatHistory collection")
            except Exception as create_error:
                if "already exists" in str(create_error).lower():
                    logger.info("ℹ️ ChatHistory collection already exists")
                else:
                    raise
            
    except Exception as e:
        logger.error(f"❌ Error initializing Weaviate collections: {e}")

def store_appointment_in_vector_db(appointment_id: int, customer_name: str, title: str, 
                                   description: str, address: str, start_time: datetime):
    """Store appointment in vector database for semantic search"""
    try:
        client = get_weaviate_client()
        if client:
            # Combine relevant fields for vectorization
            combined_text = f"Title: {title}. Description: {description}. Address: {address}. Customer: {customer_name}"
            
            # Convert datetime to RFC3339 format with timezone
            # Assume local timezone if naive datetime
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            
            appointment_collection = client.collections.get("AppointmentHistory")
            appointment_collection.data.insert({
                "appointment_id": appointment_id,
                "customer_name": customer_name.lower(),
                "title": title,
                "description": description,
                "address": address,
                "start_time": start_time.isoformat(),
                "completion_date": None,  # Will be set when appointment is completed
                "combined_text": combined_text
            })
            logger.info(f"💾 Stored appointment {appointment_id} in vector DB")
    except Exception as e:
        logger.error(f"❌ Error storing appointment in vector DB: {e}")

def find_related_appointments(customer_name: str, title: str, description: str, address: str, limit: int = 3, distance_threshold: float = 0.8) -> list:
    """Find semantically similar past appointments for a customer
    
    Args:
        customer_name: Customer's name
        title: New appointment title
        description: New appointment description
        address: New appointment address
        limit: Maximum number of related appointments to return
        distance_threshold: Maximum vector distance (0-1, lower=more similar). Default 0.8 filters generic matches.
    
    Returns:
        List of related appointments with similarity scores, sorted by relevance
    """
    try:
        client = get_weaviate_client()
        if not client:
            return []
        
        # Create enhanced search query focusing on work type and issues
        # Emphasize the description as it contains the core work details
        search_query = f"{title} {description} {address}"
        
        appointment_collection = client.collections.get("AppointmentHistory")
        
        # Search for similar appointments with distance filter
        # near_text returns certainty score (1 - distance), so convert threshold
        response = appointment_collection.query.near_text(
            query=search_query,
            limit=limit * 3,  # Get more results to filter by customer and distance
            distance=distance_threshold  # Filter out loosely related matches
        )
        
        # Filter by customer, apply distance threshold, and format results
        related = []
        for obj in response.objects:
            if obj.properties["customer_name"].lower() == customer_name.lower():
                # Calculate distance (lower is more similar)
                distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else None
                
                # Only include if distance is below threshold (more similar)
                if distance is None or distance <= distance_threshold:
                    related.append({
                        "appointment_id": obj.properties["appointment_id"],
                        "title": obj.properties["title"],
                        "description": obj.properties["description"],
                        "address": obj.properties["address"],
                        "start_time": obj.properties["start_time"],
                        "distance": distance,  # Lower is more similar
                        "similarity_score": (1 - distance) if distance is not None else None  # Convert to 0-1 score
                    })
        
        # Sort by distance (most similar first) and return top results
        # Handle None values by treating them as high distance (less similar)
        related.sort(key=lambda x: x['distance'] if x['distance'] is not None else 1.0)
        return related[:limit]
    except Exception as e:
        logger.error(f"❌ Error finding related appointments: {e}")
        return []

def sync_existing_appointments_to_vector_db():
    """One-time sync of existing appointments to vector database"""
    try:
        from calendar_agent.sql_db import get_db_connection
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id, customer_name, appointment_title, notes, address, start_time
            FROM appointments
            ORDER BY start_time
        """)
        
        appointments = cur.fetchall()
        cur.close()
        conn.close()
        
        if not appointments:
            logger.info("No appointments to sync")
            return
        
        client = get_weaviate_client()
        if not client:
            logger.error("Cannot sync - Weaviate not connected")
            return
        
        # Check if appointments already exist in vector DB
        appointment_collection = client.collections.get("AppointmentHistory")
        
        synced = 0
        for apt in appointments:
            # Check if this appointment is already in vector DB
            try:
                existing = appointment_collection.query.fetch_objects(
                    filters=Filter.by_property("appointment_id").equal(apt['id']),
                    limit=1
                )
                
                if len(existing.objects) == 0:
                    # Not in vector DB, add it
                    store_appointment_in_vector_db(
                        apt['id'],
                        apt['customer_name'],
                        apt['appointment_title'],
                        apt['notes'] or "",
                        apt['address'] or "",
                        apt['start_time']
                    )
                    synced += 1
            except Exception as e:
                logger.error(f"Error syncing appointment {apt['id']}: {e}")
                continue
        
        logger.info(f"✅ Synced {synced} appointments to vector database")
        
    except Exception as e:
        logger.error(f"❌ Error syncing appointments to vector DB: {e}")

def store_conversation(session_id: str, user_message: str, assistant_response: str):
    """Store a conversation turn in the vector database for long-term memory
    
    Args:
        session_id: Session/thread identifier
        user_message: The user's message
        assistant_response: The assistant's response
    """
    try:
        client = get_weaviate_client()
        if not client:
            return
        
        # Create combined context for better semantic search
        conversation_context = f"User asked: {user_message}. Assistant responded: {assistant_response}"
        
        chat_collection = client.collections.get("ChatHistory")
        chat_collection.data.insert({
            "session_id": session_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_context": conversation_context
        })
        logger.info(f"💾 Stored conversation in vector DB")
    except Exception as e:
        logger.error(f"❌ Error storing conversation: {e}")

def find_similar_conversations(query: str, limit: int = 5) -> list:
    """Find semantically similar past conversations
    
    Args:
        query: Search query (typically current user message)
        limit: Maximum number of similar conversations to return
    
    Returns:
        List of similar past conversations with context
    """
    try:
        client = get_weaviate_client()
        if not client:
            return []
        
        chat_collection = client.collections.get("ChatHistory")
        
        # Search for semantically similar conversations
        response = chat_collection.query.near_text(
            query=query,
            limit=limit
        )
        
        similar = []
        for obj in response.objects:
            similar.append({
                "user_message": obj.properties["user_message"],
                "assistant_response": obj.properties["assistant_response"],
                "timestamp": obj.properties["timestamp"],
                "session_id": obj.properties["session_id"]
            })
        
        return similar
    except Exception as e:
        logger.error(f"❌ Error finding similar conversations: {e}")
        return []

def get_conversation_context(query: str, max_conversations: int = 3) -> str:
    """Get relevant conversation context for the current query
    
    Args:
        query: Current user query
        max_conversations: Maximum number of past conversations to include
    
    Returns:
        Formatted string with relevant past conversation context
    """
    similar = find_similar_conversations(query, limit=max_conversations)
    
    if not similar:
        return ""
    
    context = "\n\nRelevant past conversations:\n"
    for i, conv in enumerate(similar, 1):
        context += f"{i}. Previous discussion:\n"
        context += f"   User: {conv['user_message'][:150]}...\n" if len(conv['user_message']) > 150 else f"   User: {conv['user_message']}\n"
        context += f"   Assistant: {conv['assistant_response'][:150]}...\n" if len(conv['assistant_response']) > 150 else f"   Assistant: {conv['assistant_response']}\n"
        context += f"   (From session: {conv['session_id']}, at {conv['timestamp']})\n\n"
    
    return context
