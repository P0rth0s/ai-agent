from dotenv import load_dotenv
from datetime import datetime

# Import the agent creator and tools from agent.py
from agent import create_calendar_agent

# Load environment variables
load_dotenv()

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
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nBot: Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    main()
