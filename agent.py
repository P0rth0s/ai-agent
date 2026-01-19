from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modern LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# LangGraph imports for agent with memory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

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

# Database connection helper
def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "ai_agent_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )

# Helper function for checking appointment overlaps
def check_appointment_overlap(cur, start_dt: datetime, end_dt: datetime, exclude_id: Optional[int] = None) -> Optional[dict]:
    """Check if an appointment overlaps with existing appointments.
    
    Args:
        cur: Database cursor
        start_dt: Start datetime of appointment
        end_dt: End datetime of appointment
        exclude_id: Optional appointment ID to exclude from overlap check (for updates)
    
    Returns:
        Dictionary with overlapping appointment details if overlap exists, None otherwise
    """
    if exclude_id:
        cur.execute("""
            SELECT id, appointment_title, start_time, estimated_end_time
            FROM appointments
            WHERE (start_time, estimated_end_time) OVERLAPS (%s::timestamp, %s::timestamp)
            AND id != %s
        """, (start_dt, end_dt, exclude_id))
    else:
        cur.execute("""
            SELECT id, appointment_title, start_time, estimated_end_time
            FROM appointments
            WHERE (start_time, estimated_end_time) OVERLAPS (%s::timestamp, %s::timestamp)
        """, (start_dt, end_dt))
    
    return cur.fetchone()

# Vector Database Helper Functions
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
                from datetime import timezone
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
                    filters={
                        "path": ["appointment_id"],
                        "operator": "Equal",
                        "valueInt": apt['id']
                    },
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

# Calendar Tools
@tool
def add_task(title: str, date: str, start_time: str, description: str, customer_name: str, address: str, estimated_end_time: Optional[str] = None) -> str:
    """Add an appointment to the calendar. All fields are required except estimated_end_time.
    
    Args:
        title: Appointment title (required)
        date: Date in YYYY-MM-DD format (required)
        start_time: Time in HH:MM format (required)
        description: Appointment notes/description (required)
        customer_name: Customer full name (required)
        address: Full address (required)
        estimated_end_time: Estimated end time in HH:MM format (optional, defaults to 1 hour after start)
    """
    logger.info(f"🔧 TOOL CALLED: add_task(title='{title}', date='{date}', start_time='{start_time}', customer='{customer_name}', estimated_end='{estimated_end_time}')")
    try:
        start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
        
        # Only calculate estimated_end_time if not provided
        if estimated_end_time:
            end_dt = datetime.strptime(f"{date} {estimated_end_time}", "%Y-%m-%d %H:%M")
        else:
            # Default to 1 hour duration
            end_dt = start_dt + timedelta(hours=1)
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check for related past appointments using vector search
        related_appointments = find_related_appointments(customer_name, title, description, address)
        related_info = ""
        if related_appointments:
            related_info = "\n\nℹ️ Related past appointments found:\n"
            for apt in related_appointments:
                related_info += f"  • [{apt['appointment_id']}] {apt['title']} at {apt['address']}\n"
                related_info += f"    Description: {apt['description'][:100]}...\n" if len(apt['description']) > 100 else f"    Description: {apt['description']}\n"
        
        # Check for overlapping appointments
        overlapping = check_appointment_overlap(cur, start_dt, end_dt)
        if overlapping:
            cur.close()
            conn.close()
            return f"Error: This appointment overlaps with existing appointment '{overlapping['appointment_title']}' (ID: {overlapping['id']}) scheduled from {overlapping['start_time'].strftime('%H:%M')} to {overlapping['estimated_end_time'].strftime('%H:%M')}. Please choose a different time."
        
        # Check travel time from previous appointment
        cur.execute("""
            SELECT id, appointment_title, address, estimated_end_time
            FROM appointments
            WHERE estimated_end_time <= %s
            ORDER BY estimated_end_time DESC
            LIMIT 1
        """, (start_dt,))
        previous_apt = cur.fetchone()
        
        if previous_apt and previous_apt['address']:
            travel_minutes = check_travel_time(previous_apt['address'], address)
            if travel_minutes is not None:
                time_gap = (start_dt - previous_apt['estimated_end_time']).total_seconds() / 60
                if time_gap < travel_minutes:
                    cur.close()
                    conn.close()
                    return f"Error: Not enough travel time from previous appointment '{previous_apt['appointment_title']}' (ID: {previous_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available. Consider scheduling at {(previous_apt['estimated_end_time'] + timedelta(minutes=travel_minutes)).strftime('%H:%M')} or later."
        
        # Check travel time to next appointment
        cur.execute("""
            SELECT id, appointment_title, address, start_time
            FROM appointments
            WHERE start_time >= %s
            ORDER BY start_time ASC
            LIMIT 1
        """, (end_dt,))
        next_apt = cur.fetchone()
        
        if next_apt and next_apt['address']:
            travel_minutes = check_travel_time(address, next_apt['address'])
            if travel_minutes is not None:
                time_gap = (next_apt['start_time'] - end_dt).total_seconds() / 60
                if time_gap < travel_minutes:
                    cur.close()
                    conn.close()
                    return f"Error: Not enough travel time to next appointment '{next_apt['appointment_title']}' (ID: {next_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available. Consider scheduling earlier or adjusting the end time."
        
        cur.execute("""
            INSERT INTO appointments (customer_name, address, appointment_title, notes, start_time, estimated_end_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (customer_name, address, title, description, start_dt, end_dt))
        
        appointment_id = cur.fetchone()['id']
        conn.commit()
        cur.close()
        conn.close()
        
        # Store appointment in vector database for future similarity searches
        store_appointment_in_vector_db(appointment_id, customer_name, title, description, address, start_dt)
        
        return f"✓ Appointment '{title}' scheduled for {date} at {start_time} (ID: {appointment_id}){related_info}"
    except Exception as e:
        logger.error(f"❌ Error in add_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to create appointment - {str(e)}"

@tool
def list_tasks(date: Optional[str] = None) -> str:
    """List all appointments. Optionally filter by date (YYYY-MM-DD)"""
    logger.info(f"🔧 TOOL CALLED: list_tasks(date={date})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        if date:
            cur.execute("""
                SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
                FROM appointments
                WHERE DATE(start_time) = %s
                ORDER BY start_time
            """, (date,))
        else:
            cur.execute("""
                SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
                FROM appointments
                ORDER BY start_time
            """)
        
        appointments = cur.fetchall()
        cur.close()
        conn.close()
        
        if not appointments:
            return "No appointments scheduled." if not date else f"No appointments scheduled for {date}."
        
        result = "📅 Scheduled Appointments:\n\n"
        for apt in appointments:
            result += f"[{apt['id']}] {apt['appointment_title']} - {apt['customer_name']}\n"
            result += f"    When: {apt['start_time'].strftime('%Y-%m-%d at %H:%M')} - {apt['estimated_end_time'].strftime('%H:%M')}\n"
            if apt['address']:
                result += f"    Where: {apt['address']}\n"
            if apt['notes']:
                result += f"    Notes: {apt['notes']}\n"
            result += "\n"
        return result
    except Exception as e:
        logger.error(f"❌ Error in list_tasks: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to list appointments - {str(e)}"

@tool
def delete_task(task_id: int) -> str:
    """Delete an appointment from the calendar by its ID"""
    logger.info(f"🔧 TOOL CALLED: delete_task(task_id={task_id})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT appointment_title FROM appointments WHERE id = %s", (task_id,))
        appointment = cur.fetchone()
        
        if not appointment:
            cur.close()
            conn.close()
            return f"Error: Appointment with ID {task_id} not found."
        
        cur.execute("DELETE FROM appointments WHERE id = %s", (task_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Deleted appointment: {appointment['appointment_title']} (ID: {task_id})"
    except Exception as e:
        logger.error(f"❌ Error in delete_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to delete appointment - {str(e)}"

@tool
def update_task(task_id: int, title: Optional[str] = None, date: Optional[str] = None, 
                time: Optional[str] = None, description: Optional[str] = None,
                customer_name: Optional[str] = None, address: Optional[str] = None) -> str:
    """Update an appointment's details. Provide task_id and fields to update."""
    logger.info(f"🔧 TOOL CALLED: update_task(task_id={task_id}, updates={{title={title}, date={date}, time={time}}})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get current appointment
        cur.execute("SELECT * FROM appointments WHERE id = %s", (task_id,))
        appointment = cur.fetchone()
        
        if not appointment:
            cur.close()
            conn.close()
            return f"Error: Appointment with ID {task_id} not found."
        
        # Build update query dynamically
        updates = []
        params = []
        
        if title:
            updates.append("appointment_title = %s")
            params.append(title)
        if customer_name:
            updates.append("customer_name = %s")
            params.append(customer_name)
        if address:
            updates.append("address = %s")
            params.append(address)
        if description:
            updates.append("notes = %s")
            params.append(description)
        if date or time:
            current_date = appointment['start_time'].strftime('%Y-%m-%d') if not date else date
            current_time = appointment['start_time'].strftime('%H:%M') if not time else time
            new_start = datetime.strptime(f"{current_date} {current_time}", "%Y-%m-%d %H:%M")
            # Keep same duration
            duration = appointment['estimated_end_time'] - appointment['start_time']
            new_end = new_start + duration
            
            # Check for overlapping appointments (excluding current appointment)
            overlapping = check_appointment_overlap(cur, new_start, new_end, exclude_id=task_id)
            if overlapping:
                cur.close()
                conn.close()
                return f"Error: This time change would overlap with existing appointment '{overlapping['appointment_title']}' (ID: {overlapping['id']}) scheduled from {overlapping['start_time'].strftime('%H:%M')} to {overlapping['estimated_end_time'].strftime('%H:%M')}. Please choose a different time."
            
            # Check travel time constraints when time is changing
            appointment_address = address if address else appointment['address']
            
            # Check travel time from previous appointment
            cur.execute("""
                SELECT id, appointment_title, address, estimated_end_time
                FROM appointments
                WHERE estimated_end_time <= %s AND id != %s
                ORDER BY estimated_end_time DESC
                LIMIT 1
            """, (new_start, task_id))
            previous_apt = cur.fetchone()
            
            if previous_apt and previous_apt['address'] and appointment_address:
                travel_minutes = check_travel_time(previous_apt['address'], appointment_address)
                if travel_minutes is not None:
                    time_gap = (new_start - previous_apt['estimated_end_time']).total_seconds() / 60
                    if time_gap < travel_minutes:
                        cur.close()
                        conn.close()
                        return f"Error: Not enough travel time from previous appointment '{previous_apt['appointment_title']}' (ID: {previous_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available."
            
            # Check travel time to next appointment
            cur.execute("""
                SELECT id, appointment_title, address, start_time
                FROM appointments
                WHERE start_time >= %s AND id != %s
                ORDER BY start_time ASC
                LIMIT 1
            """, (new_end, task_id))
            next_apt = cur.fetchone()
            
            if next_apt and next_apt['address'] and appointment_address:
                travel_minutes = check_travel_time(appointment_address, next_apt['address'])
                if travel_minutes is not None:
                    time_gap = (next_apt['start_time'] - new_end).total_seconds() / 60
                    if time_gap < travel_minutes:
                        cur.close()
                        conn.close()
                        return f"Error: Not enough travel time to next appointment '{next_apt['appointment_title']}' (ID: {next_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available."
            
            updates.append("start_time = %s")
            params.append(new_start)
            updates.append("estimated_end_time = %s")
            params.append(new_end)
        elif address:
            # If only address is changing (not time), check travel time constraints with current times
            # Check travel time from previous appointment
            cur.execute("""
                SELECT id, appointment_title, address, estimated_end_time
                FROM appointments
                WHERE estimated_end_time <= %s AND id != %s
                ORDER BY estimated_end_time DESC
                LIMIT 1
            """, (appointment['start_time'], task_id))
            previous_apt = cur.fetchone()
            
            if previous_apt and previous_apt['address']:
                travel_minutes = check_travel_time(previous_apt['address'], address)
                if travel_minutes is not None:
                    time_gap = (appointment['start_time'] - previous_apt['estimated_end_time']).total_seconds() / 60
                    if time_gap < travel_minutes:
                        cur.close()
                        conn.close()
                        return f"Error: Changing address would require {travel_minutes} minutes travel time from previous appointment '{previous_apt['appointment_title']}' (ID: {previous_apt['id']}), but only {int(time_gap)} minutes available."
            
            # Check travel time to next appointment
            cur.execute("""
                SELECT id, appointment_title, address, start_time
                FROM appointments
                WHERE start_time >= %s AND id != %s
                ORDER BY start_time ASC
                LIMIT 1
            """, (appointment['estimated_end_time'], task_id))
            next_apt = cur.fetchone()
            
            if next_apt and next_apt['address']:
                travel_minutes = check_travel_time(address, next_apt['address'])
                if travel_minutes is not None:
                    time_gap = (next_apt['start_time'] - appointment['estimated_end_time']).total_seconds() / 60
                    if time_gap < travel_minutes:
                        cur.close()
                        conn.close()
                        return f"Error: Changing address would require {travel_minutes} minutes travel time to next appointment '{next_apt['appointment_title']}' (ID: {next_apt['id']}), but only {int(time_gap)} minutes available."
        
        if not updates:
            cur.close()
            conn.close()
            return "No updates provided."
        
        params.append(task_id)
        query = f"UPDATE appointments SET {', '.join(updates)} WHERE id = %s RETURNING appointment_title"
        
        cur.execute(query, params)
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Updated appointment: {result['appointment_title']}"
    except Exception as e:
        logger.error(f"❌ Error in update_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to update appointment - {str(e)}"

import googlemaps

# Helper function for checking travel time
def check_travel_time(origin: str, destination: str) -> Optional[int]:
    """Check driving time between two addresses.
    
    Args:
        origin: Starting address
        destination: Ending address
    
    Returns:
        Travel time in minutes, or None if calculation fails
    """
    try:
        gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
        result = gmaps.distance_matrix(origin, destination, mode="driving")
        
        if result['rows'][0]['elements'][0]['status'] == 'OK':
            duration_seconds = result['rows'][0]['elements'][0]['duration']['value']
            duration_minutes = duration_seconds / 60
            return int(duration_minutes)
        else:
            return None
    except Exception as e:
        logger.error(f"❌ Error in check_travel_time: {type(e).__name__}: {str(e)}")
        return None

@tool
def get_previous_task(reference_time: str) -> str:
    """Get the appointment scheduled immediately before a given time.
    
    Args:
        reference_time: Time in YYYY-MM-DD HH:MM format
    """
    logger.info(f"🔧 TOOL CALLED: get_previous_task(reference_time='{reference_time}')")
    try:
        ref_dt = datetime.strptime(reference_time, "%Y-%m-%d %H:%M")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
            FROM appointments
            WHERE estimated_end_time <= %s
            ORDER BY estimated_end_time DESC
            LIMIT 1
        """, (ref_dt,))
        
        appointment = cur.fetchone()
        cur.close()
        conn.close()
        
        if not appointment:
            return "No previous appointment found."
        
        result = f"Previous Appointment:\n"
        result += f"[{appointment['id']}] {appointment['appointment_title']} - {appointment['customer_name']}\n"
        result += f"Time: {appointment['start_time'].strftime('%Y-%m-%d at %H:%M')} - {appointment['estimated_end_time'].strftime('%H:%M')}\n"
        result += f"Address: {appointment['address']}\n"
        if appointment['notes']:
            result += f"Notes: {appointment['notes']}\n"
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in get_previous_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to get previous task - {str(e)}"

@tool
def get_next_task(reference_time: str) -> str:
    """Get the appointment scheduled immediately after a given time.
    
    Args:
        reference_time: Time in YYYY-MM-DD HH:MM format
    """
    logger.info(f"🔧 TOOL CALLED: get_next_task(reference_time='{reference_time}')")
    try:
        ref_dt = datetime.strptime(reference_time, "%Y-%m-%d %H:%M")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id, customer_name, appointment_title, notes, start_time, estimated_end_time, address
            FROM appointments
            WHERE start_time >= %s
            ORDER BY start_time ASC
            LIMIT 1
        """, (ref_dt,))
        
        appointment = cur.fetchone()
        cur.close()
        conn.close()
        
        if not appointment:
            return "No next appointment found."
        
        result = f"Next Appointment:\n"
        result += f"[{appointment['id']}] {appointment['appointment_title']} - {appointment['customer_name']}\n"
        result += f"Time: {appointment['start_time'].strftime('%Y-%m-%d at %H:%M')} - {appointment['estimated_end_time'].strftime('%H:%M')}\n"
        result += f"Address: {appointment['address']}\n"
        if appointment['notes']:
            result += f"Notes: {appointment['notes']}\n"
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in get_next_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to get next task - {str(e)}"

@tool
def find_similar_appointments(customer_name: str, search_description: str) -> str:
    """Find past appointments similar to a description for a specific customer.
    
    Use this to check if a new appointment might be related to previous work.
    
    Args:
        customer_name: Customer's full name
        search_description: Description of what you're looking for (can include service type, location, issue, etc.)
    """
    logger.info(f"🔧 TOOL CALLED: find_similar_appointments(customer='{customer_name}')")
    try:
        # Use the search description as the query
        related = find_related_appointments(customer_name, search_description, search_description, "", limit=5)
        
        if not related:
            return f"No similar past appointments found for {customer_name}."
        
        result = f"📋 Similar past appointments for {customer_name}:\n\n"
        for apt in related:
            result += f"[{apt['appointment_id']}] {apt['title']}\n"
            result += f"    Address: {apt['address']}\n"
            result += f"    Description: {apt['description'][:150]}...\n" if len(apt['description']) > 150 else f"    Description: {apt['description']}\n"
            result += f"    Date: {apt['start_time']}\n\n"
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in find_similar_appointments: {e}")
        return f"Error: Failed to find similar appointments - {str(e)}"

# Setup the AI agent with memory
def create_calendar_agent():
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
    
    # Define tools
    tools = [add_task, list_tasks, delete_task, update_task, get_previous_task, get_next_task, find_similar_appointments]
    
    # Create system prompt
    system_message = f"""You are a helpful calendar assistant with long-term memory. You can help users:
    - Schedule tasks and events
    - View their calendar
    - Update or delete tasks
    - Check what's scheduled for today or specific dates
    - Find related past appointments using semantic search
    
    IMPORTANT: When scheduling a new appointment, the system will automatically check for related past appointments.
    If related appointments are found, inform the user about them as they might be relevant (follow-up work, recurring issues, etc.).
    
    You can also manually search for similar past appointments using find_similar_appointments if the user asks about history.
    
    Be conversational and helpful. When users ask to schedule something, extract the relevant details
    (title, date, time, description) and use the appropriate tool.

    Always confirm actions and provide clear feedback. Remember previous messages in our conversation.

    Today's date is {datetime.now().strftime('%Y-%m-%d')} ({datetime.now().strftime('%A, %B %d, %Y')}).
    
    IMPORTANT DATE HANDLING:
    - When users say "today", use: {datetime.now().strftime('%Y-%m-%d')}
    - When users say "tomorrow" or "tmrw", use: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
    - When users say "next week", add 7 days to today: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}
    - When users specify just a day name (e.g., "Monday"), find the next occurrence of that day
    - When users specify a date without a year, assume {datetime.now().year}
    
    IMPORTANT ADDRESS HANDLING:
    - Always confirm full addresses when scheduling tasks that require travel.
    - An Address must contain a house number, street, city, zipcode, and state.

    IMPORTANT SCHEDULING RULES:
    - Do not allow scheduling tasks in the past.
    - Do not allow scheduling overlapping tasks.
    - There must be anough travel time between the previous appointment and the new appointment, as well as between the new appointment and the next appointment.
    - If we fail any of these rules suggest alternative times based on existing calendar entries and their travel times.
    """

    # Create agent with memory checkpointer
    memory = MemorySaver()
    agent = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=system_message
    )
    
    # Sync existing appointments to vector database on startup
    sync_existing_appointments_to_vector_db()
    
    logger.info("✅ Calendar agent created with memory")
    return agent

    ### We probably want to agents actually one for the owner and one for the customer
    ### Still want to add long term memory