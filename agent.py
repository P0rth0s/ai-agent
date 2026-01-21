from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
import logging
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modern LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# LangGraph imports for agent with memory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import specialized modules
from sql_db import (
    get_db_connection, 
    check_appointment_overlap,
    get_previous_appointment,
    get_next_appointment,
    get_previous_appointment_full,
    get_next_appointment_full,
    get_all_appointments,
    get_appointment_by_id,
    insert_appointment,
    delete_appointment_by_id,
    update_appointment_fields
)
from maps import check_travel_time
from weaviate_db import (
    find_related_appointments,
    store_appointment_in_vector_db,
    sync_existing_appointments_to_vector_db,
    close_weaviate_client,
    find_similar_conversations
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
    
    Returns:
        Success message with appointment ID and any related past appointments found, or error message if scheduling conflicts exist (overlap, insufficient travel time).
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
        previous_apt = get_previous_appointment(cur, start_dt)
        
        if previous_apt and previous_apt['address']:
            travel_minutes = check_travel_time(previous_apt['address'], address)
            if travel_minutes is not None:
                time_gap = (start_dt - previous_apt['estimated_end_time']).total_seconds() / 60
                if time_gap < travel_minutes:
                    cur.close()
                    conn.close()
                    return f"Error: Not enough travel time from previous appointment '{previous_apt['appointment_title']}' (ID: {previous_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available. Consider scheduling at {(previous_apt['estimated_end_time'] + timedelta(minutes=travel_minutes)).strftime('%H:%M')} or later."
        
        # Check travel time to next appointment
        next_apt = get_next_appointment(cur, end_dt)
        
        if next_apt and next_apt['address']:
            travel_minutes = check_travel_time(address, next_apt['address'])
            if travel_minutes is not None:
                time_gap = (next_apt['start_time'] - end_dt).total_seconds() / 60
                if time_gap < travel_minutes:
                    cur.close()
                    conn.close()
                    return f"Error: Not enough travel time to next appointment '{next_apt['appointment_title']}' (ID: {next_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available. Consider scheduling earlier or adjusting the end time."
        
        appointment_id = insert_appointment(cur, customer_name, address, title, description, start_dt, end_dt)
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
    """List all appointments with complete details including ID, title, customer, time, address, and notes.
    
    Args:
        date: Optional date filter in YYYY-MM-DD format. If not provided, returns ALL appointments.
    
    Returns:
        Complete formatted list of all appointments with full details. Always present this information directly to the user without asking if they want to see more.
    """
    logger.info(f"🔧 TOOL CALLED: list_tasks(date={date})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointments = get_all_appointments(cur, date)
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
    """Delete an appointment from the calendar by its ID.
    
    Args:
        task_id: The ID of the appointment to delete
    
    Returns:
        Success message with the deleted appointment's title, or error if appointment not found.
    """
    logger.info(f"🔧 TOOL CALLED: delete_task(task_id={task_id})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointment_title = delete_appointment_by_id(cur, task_id)
        
        if not appointment_title:
            cur.close()
            conn.close()
            return f"Error: Appointment with ID {task_id} not found."
        
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Deleted appointment: {appointment_title} (ID: {task_id})"
    except Exception as e:
        logger.error(f"❌ Error in delete_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to delete appointment - {str(e)}"

@tool
def update_task(task_id: int, title: Optional[str] = None, date: Optional[str] = None, 
                time: Optional[str] = None, description: Optional[str] = None,
                customer_name: Optional[str] = None, address: Optional[str] = None) -> str:
    """Update an appointment's details. Provide task_id and fields to update.
    
    Args:
        task_id: ID of the appointment to update
        title: New appointment title (optional)
        date: New date in YYYY-MM-DD format (optional)
        time: New start time in HH:MM format (optional)
        description: New description/notes (optional)
        customer_name: New customer name (optional)
        address: New address (optional)
    
    Returns:
        Success message with updated appointment title, or error message if update conflicts exist (overlap, insufficient travel time) or appointment not found.
    """
    logger.info(f"🔧 TOOL CALLED: update_task(task_id={task_id}, updates={{title={title}, date={date}, time={time}}})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get current appointment
        appointment = get_appointment_by_id(cur, task_id)
        
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
            previous_apt = get_previous_appointment(cur, new_start, exclude_id=task_id)
            
            if previous_apt and previous_apt['address'] and appointment_address:
                travel_minutes = check_travel_time(previous_apt['address'], appointment_address)
                if travel_minutes is not None:
                    time_gap = (new_start - previous_apt['estimated_end_time']).total_seconds() / 60
                    if time_gap < travel_minutes:
                        cur.close()
                        conn.close()
                        return f"Error: Not enough travel time from previous appointment '{previous_apt['appointment_title']}' (ID: {previous_apt['id']}). Need {travel_minutes} minutes but only {int(time_gap)} minutes available."
            
            # Check travel time to next appointment
            next_apt = get_next_appointment(cur, new_end, exclude_id=task_id)
            
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
            previous_apt = get_previous_appointment(cur, appointment['start_time'], exclude_id=task_id)
            
            if previous_apt and previous_apt['address']:
                travel_minutes = check_travel_time(previous_apt['address'], address)
                if travel_minutes is not None:
                    time_gap = (appointment['start_time'] - previous_apt['estimated_end_time']).total_seconds() / 60
                    if time_gap < travel_minutes:
                        cur.close()
                        conn.close()
                        return f"Error: Changing address would require {travel_minutes} minutes travel time from previous appointment '{previous_apt['appointment_title']}' (ID: {previous_apt['id']}), but only {int(time_gap)} minutes available."
            
            # Check travel time to next appointment
            next_apt = get_next_appointment(cur, appointment['estimated_end_time'], exclude_id=task_id)
            
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
        
        result_title = update_appointment_fields(cur, task_id, updates, params)
        
        if not result_title:
            cur.close()
            conn.close()
            return f"Error: Failed to update appointment {task_id}"
        
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Updated appointment: {result_title}"
    except Exception as e:
        logger.error(f"❌ Error in update_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return f"Error: Failed to update appointment - {str(e)}"

@tool
def get_previous_task(reference_time: str) -> str:
    """Get the appointment scheduled immediately before a given time.
    
    Args:
        reference_time: Time in YYYY-MM-DD HH:MM format
    
    Returns:
        Complete details of the previous appointment including ID, title, customer, time, and address. Returns "No previous appointment found" if none exists.
    """
    logger.info(f"🔧 TOOL CALLED: get_previous_task(reference_time='{reference_time}')")
    try:
        ref_dt = datetime.strptime(reference_time, "%Y-%m-%d %H:%M")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointment = get_previous_appointment_full(cur, ref_dt)
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
    
    Returns:
        Complete details of the next appointment including ID, title, customer, time, and address. Returns "No next appointment found" if none exists.
    """
    logger.info(f"🔧 TOOL CALLED: get_next_task(reference_time='{reference_time}')")
    try:
        ref_dt = datetime.strptime(reference_time, "%Y-%m-%d %H:%M")
        
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointment = get_next_appointment_full(cur, ref_dt)
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
    
    Returns:
        List of up to 5 similar past appointments with ID, title, address, description, and date. Returns "No similar past appointments found" if none match.
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

@tool
def search_conversation_history(search_query: str) -> str:
    """Search through past conversation history to find relevant discussions.
    
    Use this when the user asks about previous conversations or when you need context from past discussions.
    
    Args:
        search_query: What to search for in past conversations (topics, questions, appointments discussed, etc.)
    
    Returns:
        List of up to 5 relevant past conversations with user messages, assistant responses, and timestamps. Returns "No relevant past conversations found" if none match.
    """
    logger.info(f"🔧 TOOL CALLED: search_conversation_history(query='{search_query}')")
    try:
        similar = find_similar_conversations(search_query, limit=5)
        
        if not similar:
            return "No relevant past conversations found."
        
        result = "📚 Relevant past conversations:\n\n"
        for i, conv in enumerate(similar, 1):
            result += f"{i}. Previous discussion:\n"
            result += f"   User: {conv['user_message'][:200]}...\n" if len(conv['user_message']) > 200 else f"   User: {conv['user_message']}\n"
            result += f"   Assistant: {conv['assistant_response'][:200]}...\n" if len(conv['assistant_response']) > 200 else f"   Assistant: {conv['assistant_response']}\n"
            result += f"   Time: {conv['timestamp']}\n\n"
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in search_conversation_history: {e}")
        return f"Error: Failed to search conversation history - {str(e)}"

# Setup the AI agent with memory
def create_calendar_agent():
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
    
    # Define tools
    tools = [add_task, list_tasks, delete_task, update_task, get_previous_task, get_next_task, 
             find_similar_appointments, search_conversation_history]
    
    # Create system prompt
    system_message = f"""You are a helpful calendar assistant with both short-term and long-term memory. You can help users:
    - Schedule tasks and events
    - View their calendar
    - Update or delete tasks
    - Check what's scheduled for today or specific dates
    - Find related past appointments using semantic search
    - Search through past conversation history
    
    MEMORY CAPABILITIES:
    - SHORT-TERM: You remember the current conversation session through your built-in memory
    - LONG-TERM: All conversations are stored in a vector database and can be searched semantically
    - Past conversations are automatically retrieved when they're relevant to the current discussion
    - Use search_conversation_history tool when the user explicitly asks about past conversations
    - When scheduling appointments, the system automatically checks for related past work
    
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