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

def find_related_appointments(customer_name: str, title: str, description: str, address: str, limit: int = 3) -> list:
    """Find semantically similar past appointments for a customer
    
    Args:
        customer_name: Customer's name
        title: New appointment title
        description: New appointment description
        address: New appointment address
        limit: Maximum number of related appointments to return
    
    Returns:
        List of related appointments with similarity scores
    """
    try:
        client = get_weaviate_client()
        if not client:
            return []
        
        # Create search query combining all relevant information
        search_query = f"Title: {title}. Description: {description}. Address: {address}"
        
        appointment_collection = client.collections.get("AppointmentHistory")
        
        # Search for similar appointments for this customer
        response = appointment_collection.query.near_text(
            query=search_query,
            limit=limit * 2  # Get more results to filter by customer
        )
        
        # Filter by customer and format results
        related = []
        for obj in response.objects:
            if obj.properties["customer_name"].lower() == customer_name.lower():
                related.append({
                    "appointment_id": obj.properties["appointment_id"],
                    "title": obj.properties["title"],
                    "description": obj.properties["description"],
                    "address": obj.properties["address"],
                    "start_time": obj.properties["start_time"],
                    "similarity_score": obj.metadata.score if hasattr(obj.metadata, 'score') else None
                })
                if len(related) >= limit:
                    break
        
        return related
    except Exception as e:
        logger.error(f"❌ Error finding related appointments: {e}")
        return []

def sync_existing_appointments_to_vector_db():
    """One-time sync of existing appointments to vector database"""
    try:
        from sql_db import get_db_connection
        
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
