from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
import logging
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modern LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# LangGraph imports for agent with memory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import specialized modules
from sql_db import get_db_connection, check_appointment_overlap
from maps import check_travel_time
from weaviate_db import (
    find_related_appointments,
    store_appointment_in_vector_db,
    sync_existing_appointments_to_vector_db,
    close_weaviate_client
)

# Load environment variables
load_dotenv()

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
    tools = [add_task, list_tasks, delete_task, update_task, get_previous_task, get_next_task, 
             find_similar_appointments]
    
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