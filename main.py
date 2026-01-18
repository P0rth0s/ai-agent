from dotenv import load_dotenv
from datetime import datetime

# Import the agent creator and tools from agent.py
from agent import create_calendar_agent

# Load environment variables
load_dotenv()

def main():
    print("🗓️  Calendar Assistant (press Ctrl+C to stop)")
    print("-" * 50)
    print("I can help you schedule tasks, view your calendar, and more!")
    print(f"Today is {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 50)
    
    # Create the calendar agent
    llm_with_tools, prompt = create_calendar_agent()
    
    # Get the tools list for execution
    from agent import add_task, list_tasks, delete_task, update_task, get_today_tasks
    tools = [add_task, list_tasks, delete_task, update_task, get_today_tasks]
    
    while True:
        try:
            # Prompt for input
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            
            # Format prompt with current date
            today = datetime.now().strftime("%Y-%m-%d")
            messages = prompt.format_messages(today=today, input=user_input)
            
            # Get response from LLM
            response = llm_with_tools.invoke(messages)
            
            # Check if tools were called
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Find and execute the tool
                    for tool in tools:
                        if tool.name == tool_name:
                            # Use invoke with the args dict
                            result = tool.invoke(input=tool_args)
                            print(f"\nBot: {result}")
                            break
            else:
                # Extract text from content
                if isinstance(response.content, list):
                    # Extract text from all text parts
                    text_parts = [part['text'] for part in response.content if part.get('type') == 'text']
                    text = ' '.join(text_parts)
                else:
                    text = response.content
                print(f"\nBot: {text}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nBot: Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
