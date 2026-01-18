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
3. Interaction with user data through a database or api. Maybe a calendar table in mysql, integrating with google calendar api would be cool if its not to hard.
4. Long term memory - Maybe with a vector database? I havent played with vector databases before so that will be extra lift.

Because of my unfamiliarity with python and AI i am going to lean into the vibe coding a bit more than i usually would.

### Overview
I successfully created an application which allows the scheduling of appoitments in a calendar stored within postgres. For short term memory of the agent I just used local memory, but this could be updated to use the same postgres database (https://docs.langchain.com/oss/python/langchain/short-term-memory).



### How to run
1. Get a gemini api key and set GOOGLE_API_KEY in your .env.example. I reccomend creating a tier 1 account which includes 300$ in credit and has a much higher rate limit than the free tier.
2. Create a google maps api key and set GOOGLE_MAPS_API_KEY in your .env.example file (https://console.cloud.google.com/google/maps-apis/credentials)
3. Copy .env.example to .env
4. Docker compose up
5. python main.py

TODO

### Testing