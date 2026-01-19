## AI Agent

#### Submitter Lee Deffebach

### Forward
I have never built an AI agent before and I havent used python since college so this was a super fun learning experience. Because of this I want to document my process going into this project.

I began by reading several dcocs

1. https://docs.langchain.com/oss/python/langchain/short-term-memory
2. https://aws.amazon.com/blogs/machine-learning/best-practices-for-building-robust-generative-ai-applications-with-amazon-bedrock-agents-part-1/
3. https://aws.amazon.com/blogs/machine-learning/best-practices-for-building-robust-generative-ai-applications-with-amazon-bedrock-agents-part-2/


At this point I had developed a few goals.

1. Play with the agent and get a better undertanding of how my prompt changes its results to the same or similar queries
2. Short term session memory - I need to better understand what session memory without a database
3. Interaction with user data through a database or api. Maybe a calendar table in mysql, integrating with a google api would be cool, maybe calendar or maps.
4. Long term memory - Maybe with a vector database? I havent played with vector databases before so that will be extra lift.

Because of my unfamiliarity with python and AI agents i am going to lean into the vibe coding a more than i usually would.

### Overview
I successfully created an application which allows the scheduling of appoitments in a calendar stored within postgres. For short term memory of the agent I just used local memory, but this could be updated to use the same postgres database (https://docs.langchain.com/oss/python/langchain/short-term-memory).

The application makes sure that appoitments do not conflict and checks that there is enough time to travel between appoitments using the google maps api.

I then added a vector database that also stores appoitment and user information. This was cool because now if i schedule appoitments for the same customer it tries to relate them together.

### Future wishes
As i worked on this project certain things began to jump out at me that id like to do but probably wouldnt have time.
1. 2 Agents - 1 for the owner of the buisness and 1 for the customer. The buisnessa agent can see all calendar information. The customer agent avoids leaking PII like names and addresses. Use of sub agents to limit context?
2. Further devloping our agent and understanding its limitations. For example i cant say delete all appoitments, but I can say delete appoitments 1, 2, and 3
3. Playing with vector db more
4. Playing with vector db more
5. More guardrails, validation, etc
6. Blocking appoitments more - We should only have appoitments between 9-5 (including travel times, implement a home base location)
5. Playing with vector db more

### How to run
1. Create and activate virtual environment: `python -m venv venv` then `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Mac/Linux)
2. Install dependencies: `pip install -r requirements.txt`
3. Get a gemini api key and set GOOGLE_API_KEY in your .env.example. I reccomend creating a tier 1 account which includes 300$ in credit and has a much higher rate limit than the free tier.
4. Create a google maps api key and set GOOGLE_MAPS_API_KEY in your .env.example file (https://console.cloud.google.com/google/maps-apis/credentials)
5. Copy .env.example to .env
6. Docker compose up
7. python main.py