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
        
        # Check for overlapping appointments
        cur.execute("""
            SELECT id, appointment_title, start_time, estimated_end_time
            FROM appointments
            WHERE (start_time, estimated_end_time) OVERLAPS (%s::timestamp, %s::timestamp)
        """, (start_dt, end_dt))
        
        overlapping = cur.fetchone()
        if overlapping:
            cur.close()
            conn.close()
            return f"Error: This appointment overlaps with existing appointment '{overlapping['appointment_title']}' (ID: {overlapping['id']}) scheduled from {overlapping['start_time'].strftime('%H:%M')} to {overlapping['estimated_end_time'].strftime('%H:%M')}. Please choose a different time."
        
        cur.execute("""
            INSERT INTO appointments (customer_name, address, appointment_title, notes, start_time, estimated_end_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (customer_name, address, title, description, start_dt, end_dt))
        
        appointment_id = cur.fetchone()['id']
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Appointment '{title}' scheduled for {date} at {start_time} (ID: {appointment_id})"
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
            updates.append("start_time = %s")
            params.append(new_start)
            # Keep same duration
            duration = appointment['estimated_end_time'] - appointment['start_time']
            updates.append("estimated_end_time = %s")
            params.append(new_start + duration)
        
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
def check_travel_time(origin: str, destination: str) -> str:
    """Check driving time between two addresses"""
    logger.info(f"🔧 TOOL CALLED: check_travel_time(origin='{origin}', dest='{destination}')")
    try:
        gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
        result = gmaps.distance_matrix(origin, destination, mode="driving")
        
        if result['rows'][0]['elements'][0]['status'] == 'OK':
            duration_seconds = result['rows'][0]['elements'][0]['duration']['value']
            duration_minutes = duration_seconds / 60
            return f"Travel time: {int(duration_minutes)} minutes"
        else:
            return "Could not calculate travel time"
    except Exception as e:
        logger.error(f"❌ Error in check_travel_time: {type(e).__name__}: {str(e)}")
        return f"Error: {str(e)}"

# Setup the AI agent with memory
def create_calendar_agent():
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
    
    # Define tools
    tools = [add_task, list_tasks, delete_task, update_task, get_previous_task, get_next_task, check_travel_time]
    
    # Create system prompt
    system_message = f"""You are a helpful calendar assistant. You can help users:
    - Schedule tasks and events
    - View their calendar
    - Update or delete tasks
    - Check what's scheduled for today or specific dates
    
    Be conversational and helpful. When users ask to schedule something, extract the relevant details
    (title, date, time, description) and use the appropriate tool.

    Always confirm actions and provide clear feedback. Remember previous messages in our conversation.

    Today's date is {datetime.now().strftime('%Y-%m-%d')} ({datetime.now().strftime('%A, %B %d, %Y')}).
    If users specify a date without a year, assume they mean this year.
    Do not allow scheduling tasks in the past.
    Do not allow scheduling overlapping tasks.

    CRITICAL: Before adding or updating any appointment, you MUST:
    1. Use get_previous_task to find the appointment ending before the new appointment
    2. If there is a previous appointment, use check_travel_time to calculate drive time from previous address to new address
    3. Verify there is at least (travel time + 15 minutes) gap between previous appointment end and new appointment start
    4. Use get_next_task to find the appointment starting after the new appointment
    5. If there is a next appointment, use check_travel_time to calculate drive time from new address to next address
    6. Verify there is at least (travel time + 15 minutes) gap between new appointment end and next appointment start
    7. Only if both checks pass (or no adjacent appointments exist), proceed with add_task or update_task
    
    If travel time checks fail, inform the user and suggest alternative times.
    """

    # Create agent with memory checkpointer
    memory = MemorySaver()
    agent = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=system_message
    )
    
    logger.info("✅ Calendar agent created with memory")
    return agent

    ### Travel time not working
    ### Asking for full date still, check address validation and parsing
    ### We probably want to agents actually one for the owner and one for the customer
    ### Still want to add long term memory