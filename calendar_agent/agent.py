from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modern LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports for agent with memory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import tools
from calendar_agent.tools import (
    add_task,
    list_tasks,
    delete_task,
    update_task,
    get_previous_task,
    get_next_task,
    find_similar_appointments,
    search_conversation_history
)

# Import database sync function
from calendar_agent.weaviate_db import sync_existing_appointments_to_vector_db

# Load environment variables
load_dotenv()

# Setup the AI agent with memory
def create_calendar_agent():
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)
    
    # Define tools
    tools = [add_task, list_tasks, delete_task, update_task, get_previous_task, get_next_task, 
             find_similar_appointments, search_conversation_history]
    
    # Create system prompt
    system_message = f"""You are a helpful calendar assistant for a handyman. You have both short-term and long-term memory. You can help users:
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
    
    IMPORTANT DATE HANDLING:
    - When users say "today", use: {datetime.now().strftime('%Y-%m-%d')}
    - When users say "tomorrow" or "tmrw", use: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
    - When users say "next week", add 7 days to today: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}
    - When users specify just a day name (e.g., "Monday"), find the next occurrence of that day
    - When users specify a date without a year, assume {datetime.now().year}
    
    IMPORTANT ADDRESS HANDLING:
    - Always confirm full addresses when scheduling tasks that require travel.
    - An Address must contain a house number, street, city, zipcode, and state.

    IMPORTANT SCHEDULING: When scheduling a new appointment, the system will automatically check for related past appointments.
    If related appointments are found, inform the user about them as they might be relevant (follow-up work, recurring issues, etc.).
    If the appointments are on the same day suggest combining them into a single visit.
    
    IMPORTANT SCHEDULING RULES:
    - Do not allow scheduling tasks in the past.
    - Do not allow scheduling overlapping tasks.
    - There must be enough travel time between the previous appointment and the new appointment, as well as between the new appointment and the next appointment.
    - Appointments are only allowed to be scheduled between 08:00 and 18:00.
    - No appointments can be scheduled on weekends (Saturday or Sunday).
    - If we fail any of these rules suggest alternative times based on existing calendar entries and their travel times.
    - All appointments must be within 2 hours of Bozeman Montana (59715). If we fail this, inform the user we are out of range.
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
