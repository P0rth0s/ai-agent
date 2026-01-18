from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os

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
def add_task(title: str, date: str, time: str = "09:00", description: str = "", customer_name: str = "Default User", address: str = "") -> str:
    """Add an appointment to the calendar. Date format: YYYY-MM-DD, Time format: HH:MM"""
    try:
        start_time = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        # Default to 1 hour duration
        estimated_end_time = start_time + timedelta(hours=1)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO appointments (customer_name, address, appointment_title, notes, start_time, estimated_end_time)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (customer_name, address, title, description, start_time, estimated_end_time))
        
        appointment_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Appointment '{title}' scheduled for {date} at {time} (ID: {appointment_id})"
    except Exception as e:
        return f"Error: Failed to create appointment - {str(e)}"

@tool
def list_tasks(date: Optional[str] = None) -> str:
    """List all appointments. Optionally filter by date (YYYY-MM-DD)"""
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
        return f"Error: Failed to list appointments - {str(e)}"

@tool
def delete_task(task_id: int) -> str:
    """Delete an appointment from the calendar by its ID"""
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
        
        return f"✓ Deleted appointment: {appointment['appointment_title']}"
    except Exception as e:
        return f"Error: Failed to delete appointment - {str(e)}"

@tool
def update_task(task_id: int, title: Optional[str] = None, date: Optional[str] = None, 
                time: Optional[str] = None, description: Optional[str] = None,
                customer_name: Optional[str] = None, address: Optional[str] = None) -> str:
    """Update an appointment's details. Provide task_id and fields to update."""
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
        return f"Error: Failed to update appointment - {str(e)}"

@tool
def get_today_tasks() -> str:
    """Get all appointments scheduled for today"""
    today = datetime.now().strftime("%Y-%m-%d")
    return list_tasks(today)

# Setup the AI agent with memory
def create_calendar_agent():
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    
    # Define tools
    tools = [add_task, list_tasks, delete_task, update_task, get_today_tasks]
    
    # Create system prompt
    system_message = f"""You are a helpful calendar assistant. You can help users:
    - Schedule tasks and events
    - View their calendar
    - Update or delete tasks
    - Check what's scheduled for today or specific dates
    
    Be conversational and helpful. When users ask to schedule something, extract the relevant details
    (title, date, time, description) and use the appropriate tool.
    
    Today's date is {datetime.now().strftime('%Y-%m-%d')}.
    
    Always confirm actions and provide clear feedback. Remember previous messages in our conversation."""
    
    # Create agent with memory checkpointer
    memory = MemorySaver()
    agent = create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        prompt=system_message
    )
    
    return agent