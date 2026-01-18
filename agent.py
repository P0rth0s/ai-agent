from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
import json

# Modern LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# In-memory calendar storage (replace with database later)
calendar_tasks = []

# Calendar Tools
@tool
def add_task(title: str, date: str, time: str = "09:00", description: str = "") -> str:
    """Add a task to the calendar. Date format: YYYY-MM-DD, Time format: HH:MM"""
    try:
        task_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        task = {
            "id": len(calendar_tasks) + 1,
            "title": title,
            "date": date,
            "time": time,
            "description": description,
            "datetime": task_datetime.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        calendar_tasks.append(task)
        return f"✓ Task '{title}' scheduled for {date} at {time}"
    except ValueError as e:
        return f"Error: Invalid date/time format. Use YYYY-MM-DD for date and HH:MM for time."

@tool
def list_tasks(date: Optional[str] = None) -> str:
    """List all calendar tasks. Optionally filter by date (YYYY-MM-DD)"""
    if not calendar_tasks:
        return "No tasks scheduled."
    
    filtered_tasks = calendar_tasks
    if date:
        filtered_tasks = [t for t in calendar_tasks if t["date"] == date]
        if not filtered_tasks:
            return f"No tasks scheduled for {date}."
    
    result = "📅 Scheduled Tasks:\n\n"
    for task in sorted(filtered_tasks, key=lambda x: x["datetime"]):
        result += f"[{task['id']}] {task['title']}\n"
        result += f"    When: {task['date']} at {task['time']}\n"
        if task['description']:
            result += f"    Note: {task['description']}\n"
        result += "\n"
    return result

@tool
def delete_task(task_id: int) -> str:
    """Delete a task from the calendar by its ID"""
    global calendar_tasks
    for i, task in enumerate(calendar_tasks):
        if task["id"] == task_id:
            removed = calendar_tasks.pop(i)
            return f"✓ Deleted task: {removed['title']}"
    return f"Error: Task with ID {task_id} not found."

@tool
def update_task(task_id: int, title: Optional[str] = None, date: Optional[str] = None, 
                time: Optional[str] = None, description: Optional[str] = None) -> str:
    """Update a task's details. Provide task_id and fields to update."""
    for task in calendar_tasks:
        if task["id"] == task_id:
            if title:
                task["title"] = title
            if date:
                task["date"] = date
            if time:
                task["time"] = time
            if description:
                task["description"] = description
            if date or time:
                task["datetime"] = datetime.strptime(
                    f"{task['date']} {task['time']}", "%Y-%m-%d %H:%M"
                ).isoformat()
            return f"✓ Updated task: {task['title']}"
    return f"Error: Task with ID {task_id} not found."

@tool
def get_today_tasks() -> str:
    """Get all tasks scheduled for today"""
    today = datetime.now().strftime("%Y-%m-%d")
    return list_tasks(today)

# Setup the AI agent
def create_calendar_agent():
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)
    
    # Define tools
    tools = [add_task, list_tasks, delete_task, update_task, get_today_tasks]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create system prompt
    system_prompt = """You are a helpful calendar assistant. You can help users:
    - Schedule tasks and events
    - View their calendar
    - Update or delete tasks
    - Check what's scheduled for today or specific dates
    
    Be conversational and helpful. When users ask to schedule something, extract the relevant details
    (title, date, time, description) and use the appropriate tool.
    
    Today's date is {today}.
    
    Always confirm actions and provide clear feedback."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    return llm_with_tools, prompt