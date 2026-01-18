# Vector Memory Features

## Overview
The calendar agent now has long-term memory capabilities using Weaviate vector database. This allows the agent to remember conversations, customer preferences, and appointment outcomes across sessions.

## Features Implemented

### 1. Chat History Storage
- **Automatic**: Every conversation is automatically stored in the vector database
- **Semantic Search**: Past conversations can be retrieved based on meaning, not just keywords
- **Persistent**: Survives application restarts

### 2. Customer Preferences
The agent can learn and remember customer preferences such as:
- Preferred appointment times (e.g., "John prefers morning appointments")
- Communication preferences (e.g., "Sarah likes text reminders")
- Service preferences (e.g., "Mike always needs 2-hour slots")
- Special requirements (e.g., "Needs gate code at delivery")

### 3. Appointment Outcomes
Track what happened during appointments:
- Service details and work performed
- Issues found and resolved
- Parts used or needed
- Follow-up requirements
- Customer feedback

## Available Tools

### `get_customer_context(customer_name)`
Retrieves all historical information about a customer including:
- Stored preferences
- Past appointment notes
- Follow-up requirements

**Example Usage:**
```
You: Get me information about John Smith
Bot: [Retrieves John's preferences and appointment history]
```

### `remember_customer_preference(customer_name, preference_type, preference_details)`
Manually store a customer preference for future reference.

**Example Usage:**
```
You: Remember that John Smith prefers morning appointments between 8-10am
Bot: [Agent uses the tool to store this preference]
```

### `save_appointment_outcome(appointment_id, outcome_notes, follow_up_needed, follow_up_date)`
Save detailed notes after completing an appointment.

**Example Usage:**
```
You: Save outcome for appointment 5: Replaced water heater, customer needs follow-up in 6 months
Bot: [Stores outcome with follow-up flag]
```

## How It Works

### Automatic Learning
The agent is instructed to:
1. **Always check customer context** when scheduling for returning customers
2. **Learn from conversations** - if a customer mentions preferences, save them
3. **Encourage outcome recording** after appointments are completed

### Vector Embeddings
- Text is converted to vector embeddings using Weaviate's text2vec-transformers
- Enables semantic search: "preferred times" will match "likes mornings"
- Finds relevant information even with different wording

### Data Structure

**ChatHistory Collection:**
- user_message
- assistant_response
- timestamp
- session_id

**CustomerPreferences Collection:**
- customer_name
- preference_type (timing, communication, service)
- preference_details
- learned_date
- confidence (how many times observed)

**AppointmentOutcomes Collection:**
- appointment_id
- customer_name
- service_type
- outcome_notes
- completion_date
- follow_up_needed
- follow_up_date

## Example Workflows

### Scheduling for a Returning Customer
```
You: Schedule an appointment for John Smith
Bot: [Automatically checks get_customer_context]
Bot: I see John Smith prefers morning appointments between 8-10am. 
     Would you like to schedule at 9:00am tomorrow?
```

### Learning Preferences
```
You: Schedule Sarah at 3pm tomorrow, she mentioned she prefers afternoon slots
Bot: [Schedules appointment and stores preference]
Bot: ✓ Appointment scheduled. I'll remember Sarah prefers afternoon appointments.
```

### Recording Outcomes
```
You: Mark appointment 12 as complete. We replaced the furnace filter and 
     noticed the thermostat needs replacement. Schedule follow-up in 3 months.
Bot: [Uses save_appointment_outcome tool]
Bot: ✓ Outcome saved with follow-up needed in 3 months.
```

### Using Historical Context
```
You: What did we do for Mike Johnson last time?
Bot: [Retrieves appointment history]
Bot: Last time we serviced Mike's HVAC system and noted the compressor 
     was showing early signs of wear. That was 2 months ago.
```

## Technical Details

### Prerequisites
- Weaviate instance running (configured in docker-compose.yml)
- WEAVIATE_HOST and WEAVIATE_PORT environment variables

### Collections Initialized Automatically
Collections are created on first connection to Weaviate with appropriate schemas and vectorizers.

### Performance Considerations
- Vector similarity search is fast even with thousands of records
- Limit results to relevant recent items
- Consider periodic cleanup of old chat history

## Future Enhancements
- Automatic preference detection from patterns
- Sentiment analysis on customer interactions
- Predictive scheduling suggestions
- Integration with customer feedback
