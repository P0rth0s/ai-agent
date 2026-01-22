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

# Import tools from calendar_agent package
from calendar_agent.tools.calendar_tools import add_task, list_tasks, delete_task, update_task
from calendar_agent.tools.query_tools import get_previous_task, get_next_task
from calendar_agent.tools.search_tools import find_similar_appointments, search_conversation_history

# Import specialized modules for startup sync
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