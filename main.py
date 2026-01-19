from dotenv import load_dotenv
from datetime import datetime
import re
import time
import logging

# Import the agent creator and tools from agent.py
from agent import create_calendar_agent
from weaviate_db import close_weaviate_client
import atexit

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register cleanup function to close Weaviate connection on exit
atexit.register(close_weaviate_client)

def format_error(error):
    """Extract and format the main error message"""
    error_str = str(error)
    
    # Try to extract the 'message' field from API errors
    message_match = re.search(r"'message': '([^']+)'", error_str)
    if message_match:
        return message_match.group(1)
    
    # Otherwise return the full error
    return error_str

def extract_retry_delay(error):
    """Extract retry delay in seconds from error message"""
    error_str = str(error)
    # Try to extract from "Please retry in X.XXXs"
    retry_match = re.search(r'retry in ([\d.]+)s', error_str)
    if retry_match:
        return float(retry_match.group(1))
    return None

def is_rate_limit_error(error):
    """Check if error is a 429 rate limit error"""
    error_str = str(error)
    return '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str

def main():
    print("🗓️  Calendar Assistant with Memory (press Ctrl+C to stop)")
    print("-" * 50)
    print("I can help you schedule tasks, view your calendar, and more!")
    print(f"Today is {datetime.now().strftime('%Y-%m-%d')}")
    print("I'll remember our conversation!")
    print("-" * 50)
    
    # Create the calendar agent with memory
    agent = create_calendar_agent()
    
    # Thread ID for session memory
    thread_id = "user_session_1"
    
    while True:
        try:
            # Prompt for input
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            # Retry logic for rate limit errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"💬 User input: {user_input}")
                    # Invoke agent with memory using thread_id
                    response = agent.invoke(
                        {"messages": [("user", user_input)]},
                        config={"configurable": {"thread_id": thread_id}}
                    )
                    
                    # Extract and print the final response
                    if response and "messages" in response:
                        last_message = response["messages"][-1]
                        if hasattr(last_message, 'content'):
                            content = last_message.content
                            # Extract text if it's a list
                            if isinstance(content, list):
                                text_parts = [part['text'] for part in content if isinstance(part, dict) and part.get('type') == 'text']
                                content = ' '.join(text_parts) if text_parts else str(content)
                            print(f"\nBot: {content}")
                        else:
                            print(f"\nBot: {last_message}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    # Check if it's a rate limit error
                    if is_rate_limit_error(e):
                        retry_delay = extract_retry_delay(e)
                        if retry_delay and attempt < max_retries - 1:
                            print(f"\n⏳ Rate limit reached. Waiting {int(retry_delay)} seconds before retrying...")
                            time.sleep(retry_delay)
                            continue
                    # If not a rate limit error or last attempt, raise to outer handler
                    raise
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            close_weaviate_client()  # Explicitly close connection on exit
            break
        except Exception as e:
            formatted_error = format_error(e)
            print(f"\nBot: {formatted_error}")

if __name__ == "__main__":
    main()
