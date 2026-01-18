# Load environment variables from a .env file.
from dotenv import load_dotenv

# Define structured output models using Pydantic
from pydantic import BaseModel

# Langchain imports that we will use to interact with Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Custom tools that we will use. These are pulled from our tools.py
from tools import scrape_tool, search_tool, save_tool  

# Pulling our Gemini API key from our .env file.
load_dotenv()

# Define the structure of each lead in the output
class LeadResponse(BaseModel):
    company: str
    contact_info: str
    email: str
    summary: str
    outreach_message: str
    tools_used: list[str]

# Define a list structure to hold multiple leads
class LeadResponseList(BaseModel):
    leads: list[LeadResponse]

# Determining which AI model we will use, in this case, Gemini-2.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Tell Gemini how to format the response using the Pydantic schema
parser = PydanticOutputParser(pydantic_object=LeadResponseList)

# The main part here. This is our prompt and the instructions we give to Gemini
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a sales enablement assistant.
            1. Use the 'scrape' tool to find exactly 5 local small businesses in Vancouver, British Columbia, from a variety of industries, that might need IT services.
            2. For each company identified by the 'scrape' tool, use the 'search' tool to gather detailed information from DuckDuckGo.
            3. Analyze the searched website content to provide:
                - company: The company name
                - contact_info: Any available contact details
                - summary: A brief qualification based on the scraped website content, focusing on their potential IT needs even if they are not an IT company.
                - email addresses
                - outreach message
                - tools_used: List tools used        

            Do not include extra text beyond the formatted output and the save confirmation message.
            4. Return the output as a list of 5 entries in this format: {format_instructions}
            5. After formatting the list of 5 entries, use the 'save_to_text' tool to send the json format to the text file. 
            6. If the 'save' tool runs, say that you ran it. If you did not run the 'save' tool, say that you could not run it.
            """,
        ),
        ("human", "{query}"),  # The actual user instruction
        ("placeholder", "{agent_scratchpad}"),  # Where the agent's internal reasoning goes
    ]
).partial(format_instructions=parser.get_format_instructions())

# List the tools we are telling our LLM to use from our tools.py file
tools = [scrape_tool, search_tool, save_tool]

# Bind tools to the LLM - modern LangChain approach
llm_with_tools = llm.bind_tools(tools)

# Create a simple agent chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentFinish

def run_agent(user_input):
    agent_input = {"query": user_input, "agent_scratchpad": ""}
    
    # Format the prompt
    formatted_prompt = prompt.format_messages(**agent_input)
    
    # Invoke the LLM with tools
    response = llm_with_tools.invoke(formatted_prompt)
    
    return response.content

# Define the query that kicks off the lead generation process
query = "Find and qualify exactly 5 local leads in Vancouver for IT Services. No more than 5 small businesses."

# Run the agent with the query
raw_response = run_agent(query)

# Parse the structured output using the Pydantic schema
try:
    structured_response = parser.parse(raw_response)
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)