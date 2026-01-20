from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
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
from maps import check_travel_time, validate_travel_time_from_previous, validate_travel_time_to_next
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
def add_task(title: str, date: str, start_time: str, description: str, customer_name: str, address: str, estimated_end_time: Optional[str] = None) -> Dict[str, Any]:
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
        Dictionary with:
        - success (bool): True if appointment created, False if error
        - appointment_id (int): ID of created appointment (only if success=True)
        - title (str): Title of the appointment (only if success=True)
        - date (str): Date of appointment (only if success=True)
        - time (str): Start time of appointment (only if success=True)
        - error (str): Error message (only if success=False)
        - related_appointments (list): List of related past appointments with fields: appointment_id, title, address, description (only if success=True and related found)
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
            return {
                "success": False,
                "error": f"This appointment overlaps with existing appointment '{overlapping['appointment_title']}' (ID: {overlapping['id']}) scheduled from {overlapping['start_time'].strftime('%H:%M')} to {overlapping['estimated_end_time'].strftime('%H:%M')}. Please choose a different time."
            }
        
        # Check travel time from previous appointment
        previous_apt = get_previous_appointment(cur, start_dt)
        travel_error = validate_travel_time_from_previous(previous_apt, start_dt, address)
        if travel_error:
            cur.close()
            conn.close()
            return {"success": False, **travel_error}
        
        # Check travel time to next appointment
        next_apt = get_next_appointment(cur, end_dt)
        travel_error = validate_travel_time_to_next(next_apt, end_dt, address)
        if travel_error:
            cur.close()
            conn.close()
            return {"success": False, **travel_error}
        
        appointment_id = insert_appointment(cur, customer_name, address, title, description, start_dt, end_dt)
        conn.commit()
        cur.close()
        conn.close()
        
        # Store appointment in vector database for future similarity searches
        store_appointment_in_vector_db(appointment_id, customer_name, title, description, address, start_dt)
        
        result = {
            "success": True,
            "appointment_id": appointment_id,
            "title": title,
            "date": date,
            "time": start_time
        }
        
        if related_appointments:
            result["related_appointments"] = related_appointments
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in add_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to create appointment - {str(e)}"}

@tool
def list_tasks(date: Optional[str] = None) -> Dict[str, Any]:
    """List all appointments with complete details including ID, title, customer, time, address, and notes.
    
    Args:
        date: Optional date filter in YYYY-MM-DD format. If not provided, returns ALL appointments.
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - appointments (list): List of appointment dictionaries, each with fields: id, appointment_title, customer_name, start_time, estimated_end_time, address, notes
        - date_filter (str): The date filter used, if any
        - count (int): Number of appointments returned
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: list_tasks(date={date})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointments = get_all_appointments(cur, date)
        cur.close()
        conn.close()
        
        # Convert datetime objects to strings for JSON serialization
        appointments_list = []
        for apt in appointments:
            appointments_list.append({
                "id": apt['id'],
                "appointment_title": apt['appointment_title'],
                "customer_name": apt['customer_name'],
                "start_time": apt['start_time'].strftime('%Y-%m-%d %H:%M'),
                "estimated_end_time": apt['estimated_end_time'].strftime('%Y-%m-%d %H:%M'),
                "address": apt['address'],
                "notes": apt['notes']
            })
        
        result = {
            "success": True,
            "appointments": appointments_list,
            "count": len(appointments_list)
        }
        
        if date:
            result["date_filter"] = date
        
        return result
    except Exception as e:
        logger.error(f"❌ Error in list_tasks: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to list appointments - {str(e)}"}

@tool
def delete_task(task_id: int) -> Dict[str, Any]:
    """Delete an appointment from the calendar by its ID.
    
    Args:
        task_id: The ID of the appointment to delete
    
    Returns:
        Dictionary with:
        - success (bool): True if deleted, False if error
        - task_id (int): ID of the deleted appointment (only if success=True)
        - title (str): Title of deleted appointment (only if success=True)
        - error (str): Error message (only if success=False)
    """
    logger.info(f"🔧 TOOL CALLED: delete_task(task_id={task_id})")
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        appointment_title = delete_appointment_by_id(cur, task_id)
        
        if not appointment_title:
            cur.close()
            conn.close()
            return {"success": False, "error": f"Appointment with ID {task_id} not found."}
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"success": True, "task_id": task_id, "title": appointment_title}
    except Exception as e:
        logger.error(f"❌ Error in delete_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to delete appointment - {str(e)}"}

@tool
def update_task(task_id: int, title: Optional[str] = None, date: Optional[str] = None, 
                time: Optional[str] = None, description: Optional[str] = None,
                customer_name: Optional[str] = None, address: Optional[str] = None) -> Dict[str, Any]:
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
        Dictionary with:
        - success (bool): True if updated, False if error
        - task_id (int): ID of the updated appointment (only if success=True)
        - title (str): Title of updated appointment (only if success=True)
        - error (str): Error message (only if success=False)
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
            return {"success": False, "error": f"Appointment with ID {task_id} not found."}
        
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
                return {
                    "success": False,
                    "error": f"This time change would overlap with existing appointment '{overlapping['appointment_title']}' (ID: {overlapping['id']}) scheduled from {overlapping['start_time'].strftime('%H:%M')} to {overlapping['estimated_end_time'].strftime('%H:%M')}. Please choose a different time."
                }
            
            # Check travel time constraints when time is changing
            appointment_address = address if address else appointment['address']
            
            # Check travel time from previous appointment
            previous_apt = get_previous_appointment(cur, new_start, exclude_id=task_id)
            travel_error = validate_travel_time_from_previous(previous_apt, new_start, appointment_address)
            if travel_error:
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
            
            # Check travel time to next appointment
            next_apt = get_next_appointment(cur, new_end, exclude_id=task_id)
            travel_error = validate_travel_time_to_next(next_apt, new_end, appointment_address)
            if travel_error:
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
            
            updates.append("start_time = %s")
            params.append(new_start)
            updates.append("estimated_end_time = %s")
            params.append(new_end)
        elif address:
            # If only address is changing (not time), check travel time constraints with current times
            # Check travel time from previous appointment
            previous_apt = get_previous_appointment(cur, appointment['start_time'], exclude_id=task_id)
            travel_error = validate_travel_time_from_previous(previous_apt, appointment['start_time'], address)
            if travel_error:
                # Adjust error message for address change context
                travel_error['error'] = travel_error['error'].replace('Not enough travel time', 'Changing address would require')
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
            
            # Check travel time to next appointment
            next_apt = get_next_appointment(cur, appointment['estimated_end_time'], exclude_id=task_id)
            travel_error = validate_travel_time_to_next(next_apt, appointment['estimated_end_time'], address)
            if travel_error:
                # Adjust error message for address change context
                travel_error['error'] = travel_error['error'].replace('Not enough travel time', 'Changing address would require')
                cur.close()
                conn.close()
                return {"success": False, **travel_error}
        
        if not updates:
            cur.close()
            conn.close()
            return {"success": False, "error": "No updates provided."}
        
        result_title = update_appointment_fields(cur, task_id, updates, params)
        
        if not result_title:
            cur.close()
            conn.close()
            return {"success": False, "error": f"Failed to update appointment {task_id}"}
        
        conn.commit()
        cur.close()
        conn.close()
        
        return {"success": True, "task_id": task_id, "title": result_title}
    except Exception as e:
        logger.error(f"❌ Error in update_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to update appointment - {str(e)}"}

@tool
def get_previous_task(reference_time: str) -> Dict[str, Any]:
    """Get the appointment scheduled immediately before a given time.
    
    Args:
        reference_time: Time in YYYY-MM-DD HH:MM format
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - appointment (dict): Appointment data with fields: id, appointment_title, customer_name, start_time, estimated_end_time, address, notes (only if found)
        - error (str): Error message (only if success=False)
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
            return {"success": True, "appointment": None}
        
        return {
            "success": True,
            "appointment": {
                "id": appointment['id'],
                "appointment_title": appointment['appointment_title'],
                "customer_name": appointment['customer_name'],
                "start_time": appointment['start_time'].strftime('%Y-%m-%d %H:%M'),
                "estimated_end_time": appointment['estimated_end_time'].strftime('%Y-%m-%d %H:%M'),
                "address": appointment['address'],
                "notes": appointment['notes']
            }
        }
    except Exception as e:
        logger.error(f"❌ Error in get_previous_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to get previous task - {str(e)}"}

@tool
def get_next_task(reference_time: str) -> Dict[str, Any]:
    """Get the appointment scheduled immediately after a given time.
    
    Args:
        reference_time: Time in YYYY-MM-DD HH:MM format
    
    Returns:
        Dictionary with:
        - success (bool): True if query succeeded, False if error
        - appointment (dict): Appointment data with fields: id, appointment_title, customer_name, start_time, estimated_end_time, address, notes (only if found)
        - error (str): Error message (only if success=False)
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
            return {"success": True, "appointment": None}
        
        return {
            "success": True,
            "appointment": {
                "id": appointment['id'],
                "appointment_title": appointment['appointment_title'],
                "customer_name": appointment['customer_name'],
                "start_time": appointment['start_time'].strftime('%Y-%m-%d %H:%M'),
                "estimated_end_time": appointment['estimated_end_time'].strftime('%Y-%m-%d %H:%M'),
                "address": appointment['address'],
                "notes": appointment['notes']
            }
        }
    except Exception as e:
        logger.error(f"❌ Error in get_next_task: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return {"success": False, "error": f"Failed to get next task - {str(e)}"}

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
    - Search for similar past appointments
    
    Be conversational and helpful. When users ask to schedule something, extract the relevant details
    (title, date, time, description) and use the appropriate tool.

    Always confirm actions and provide clear feedback. Remember previous messages in our conversation.

    Today's date is {datetime.now().strftime('%Y-%m-%d')} ({datetime.now().strftime('%A, %B %d, %Y')}).

    TOOL RESPONSES:
    - All tools return structured dictionary data, NOT user-friendly text
    - You MUST format tool responses into natural, conversational messages for users
    - When listing appointments, format them clearly with IDs in brackets like: [1] Meeting - John Doe
    - When showing multiple items, use appropriate formatting (bullets, numbers, line breaks)
    - Extract and present relevant information from the structured data
    - For operations like "delete all appointments", you can call list_tasks to get all appointment IDs, then call delete_task multiple times

    MEMORY CAPABILITIES:
    - SHORT-TERM: You remember the current conversation session through your built-in memory
    - LONG-TERM: All conversations are stored in a vector database and can be searched semantically
    - Past conversations are automatically retrieved when they're relevant to the current discussion
    - Use search_conversation_history tool when the user explicitly asks about past conversations
    - When scheduling appointments, the system automatically checks for related past work
    
    IMPORTANT: When scheduling a new appointment, the system will automatically check for related past appointments.
    If related appointments are found, inform the user about them as they might be relevant (follow-up work, recurring issues, etc.).
    If the appointments are on the same day suggest combining them into a single visit.
    
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