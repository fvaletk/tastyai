# TastyAI Architecture Document

## Overview

TastyAI is an intelligent recipe recommendation system that uses a multi-agent architecture to understand user preferences, search through a recipe database, and provide personalized meal suggestions. The system is built with a focus on modularity, conversational flow, and multilingual support.

## Agentic Architecture

### Design Philosophy

The system follows a **modular, agent-based architecture** where each agent has a specific responsibility. This design choice allows for:

- **Separation of concerns**: Each agent handles one aspect of the problem
- **Easier debugging**: Issues can be isolated to specific agents
- **Scalability**: Agents can be improved or replaced independently
- **Maintainability**: Clear boundaries make the codebase easier to understand

### Architecture Flow

The system uses **LangGraph** to orchestrate a state machine that routes user requests through specialized agents:

```
User Input
    ↓
[Intent Classification Agent]
    ↓
    ├─→ new_search → [Preference Parser Agent] → [Recipe Search Agent] → [Response Agent]
    ├─→ comparison → [Response Agent] (reuses existing results)
    ├─→ recipe_request → [Recipe Request Analysis Agent] → [Response Agent]
    └─→ general → [Response Agent] (reuses existing results)
```

### Core Components

#### 1. Intent Classification Agent (`conversation_agent.py`)

**Purpose**: Determines what the user wants to do based on conversation context.

**Technology**: OpenAI GPT-4o

**Responsibilities**:
- Analyzes the user's latest message in context of conversation history
- Classifies intent into one of four categories:
  - `new_search`: User wants to search for new recipes
  - `comparison`: User is comparing previously shown recipes
  - `recipe_request`: User wants the full recipe for a specific dish
  - `general`: General questions or conversation

**Key Design Decision**: Uses GPT-4o for intent classification to ensure high accuracy in understanding user intent and conversation context, which is critical for proper routing through the agent workflow.

#### 2. Preference Parser Agent (`parser_agent.py`)

**Purpose**: Extracts structured preferences from unstructured user input.

**Technology**: OpenAI GPT-4o with function calling

**Responsibilities**:
- Parses natural language to extract:
  - Language (for multilingual support)
  - Cuisine type
  - Dietary restrictions
  - Specific dish
  - Desired ingredients
  - Allergies to avoid
  - Meal type
  - Cooking time preferences

**Key Design Decision**: Uses OpenAI's function calling feature to ensure structured output. This is more reliable than asking the model to return JSON and parsing it, as function calling guarantees valid schema compliance.

#### 3. Recipe Search Agent (`search_agent.py`)

**Purpose**: Performs semantic search in the recipe database.

**Technology**: 
- OpenAI `text-embedding-3-small` for embeddings
- Pinecone vector database for similarity search

**Responsibilities**:
- Converts user preferences into a query string
- Generates embeddings using OpenAI's embedding model
- Searches Pinecone for top 3 matching recipes
- Returns recipe metadata (title, ingredients, directions, source)

**Key Design Decision**: 
- Uses semantic search rather than keyword matching, allowing the system to understand intent (e.g., "quick dinner" matches recipes with short cooking times)
- Top-k of 3 recipes balances variety with relevance
- `text-embedding-3-small` provides good quality at a lower cost than larger embedding models

#### 4. Recipe Request Analysis Agent (`recipe_request_agent.py`)

**Purpose**: Analyzes recipe requests to determine if user wants a specific recipe, a dish type, or a new search.

**Technology**: OpenAI GPT-4o-mini

**Responsibilities**:
- Extracts recipe titles from previous conversation messages
- Determines if user is requesting:
  - A specific recipe (by name or ordinal reference like "the first one")
  - A dish type that was shown before
  - A completely new dish
- Handles ordinal references ("the second option", "the first one") by mapping them to previously shown recipes

**Key Design Decision**: This agent prevents unnecessary re-searches when users are referring to recipes already shown. It uses regex patterns to extract recipe titles from markdown-formatted assistant messages, then uses an LLM to understand user intent.

#### 5. Response Generation Agent (`response_agent.py`)

**Purpose**: Generates natural, conversational responses based on intent and context.

**Technology**: OpenAI GPT-4o

**Responsibilities**:
- Routes to specialized response handlers based on intent:
  - `handle_new_search`: Presents recipe options without full details
  - `handle_recipe_request`: Returns full recipe with ingredients and directions
  - `handle_comparative_response`: Compares recipes based on user questions
  - `handle_general_response`: Answers general questions about recipes
- Maintains multilingual consistency (responds in the user's detected language)
- Formats recipes in a readable way

**Key Design Decision**: Uses GPT-4o for response generation to ensure high-quality, natural language output. Different handlers for different intents allow for specialized prompting and better user experience.

## Technology Stack

### Core Framework
- **LangGraph**: Orchestrates the agent workflow using a state machine pattern
- **FastAPI**: RESTful API backend for handling HTTP requests
- **Streamlit**: Web UI for user interaction

### AI/ML Models
- **OpenAI GPT-4o**: Primary LLM for parsing and response generation
- **OpenAI GPT-4o-mini**: Lightweight model for classification tasks
- **OpenAI text-embedding-3-small**: Embedding model for semantic search

### Databases
- **Pinecone**: Vector database for semantic recipe search
- **PostgreSQL**: Relational database for conversation history (via SQLAlchemy)

### Supporting Libraries
- **LangChain**: Tool definitions and integrations
- **Pydantic**: Data validation and schema definitions
- **SQLAlchemy**: ORM for database operations
- **Alembic**: Database migrations

## Data Flow

### Request Processing

1. **User sends message** → FastAPI endpoint receives request
2. **Load conversation history** → PostgreSQL database queried for previous messages
3. **Initialize state** → LangGraph state object created with user input and history
4. **Intent classification** → First agent determines user intent
5. **Conditional routing** → Based on intent:
   - New search: Parse → Search → Response
   - Comparison/General: Direct to Response (reuse results)
   - Recipe request: Analyze request → Route accordingly
6. **Response generation** → Final agent generates natural language response
7. **Save to database** → Both user and assistant messages saved to PostgreSQL
8. **Return to user** → Response sent back through FastAPI

### State Management

The system uses LangGraph's state management to pass data between agents:

```python
class TastyAIState:
    user_input: str
    conversation_id: str
    preferences: UserPreferences
    results: List[RecipeMatch]
    previous_results: List[RecipeMatch]  # For context in follow-ups
    generated_response: str
    intent: str
    messages: List[Message]
    # ... additional routing fields
```

**Key Design Decision**: Storing `previous_results` in state allows the system to maintain context across multiple turns, so users can refer back to recipes shown earlier in the conversation.

## Database Design

### PostgreSQL Schema

**Table: `chat_messages`**
- `id`: Primary key
- `conversation_id`: Groups messages by conversation
- `role`: "user" or "assistant"
- `content`: Message text
- `timestamp`: Auto-generated timestamp

**Purpose**: Persists conversation history for context-aware responses and allows users to resume conversations.

### Pinecone Index

**Structure**: Vector embeddings of recipe metadata
- Each recipe is embedded with its title, ingredients, directions, cuisine, etc.
- Metadata includes: title, link, ingredients (list), directions (list), source

**Purpose**: Enables semantic search - finding recipes based on meaning rather than exact keyword matches.

## Methodology

### Problem Breakdown

Following the technical specifications, the problem was broken down into specialized components:

1. **Preference Extraction**: Convert natural language to structured data
2. **Recipe Search**: Find relevant recipes from a large dataset
3. **Response Generation**: Create natural, helpful responses
4. **Conversation Management**: Maintain context across multiple turns
5. **Intent Understanding**: Route requests to appropriate handlers

### Modular Design Benefits

Each agent is a separate module with clear inputs and outputs:

- **Testability**: Each agent can be tested independently
- **Debugging**: Issues can be traced to specific agents
- **Iteration**: Improvements to one agent don't affect others
- **Reusability**: Agents can be reused in different contexts

### Challenges and Solutions

#### Challenge 1: Maintaining Context Across Conversations

**Problem**: Users might refer to recipes shown earlier in the conversation.

**Solution**: 
- Store `previous_results` in state
- Recipe Request Analysis Agent extracts recipe titles from conversation history
- Response Agent can access both current and previous results

#### Challenge 2: Handling Ordinal References

**Problem**: Users say "the first one" or "the second option" without specifying recipe names.

**Solution**:
- Recipe Request Analysis Agent uses regex to extract numbered recipe lists from assistant messages
- Maps ordinal references to actual recipe titles
- Handles edge cases like "that one" by analyzing conversation context

#### Challenge 3: Avoiding Redundant Searches

**Problem**: Follow-up questions shouldn't trigger new searches if user is discussing existing results.

**Solution**:
- Intent Classification Agent distinguishes between new searches and follow-ups
- Conditional routing skips search for comparison/general intents
- Previous results are preserved in state

#### Challenge 4: Multilingual Support

**Problem**: Users might switch languages or use different languages in the same conversation.

**Solution**:
- Preference Parser Agent detects language from user input
- Language is stored in preferences and passed to Response Agent
- Response Agent generates responses in the detected language

#### Challenge 5: Recipe Matching Accuracy

**Problem**: Matching user requests to specific recipes when recipe titles might not match exactly.

**Solution**:
- Recipe Request Analysis Agent uses multiple matching strategies:
  - Exact match (case-insensitive)
  - Partial match (substring)
  - Fuzzy match (word overlap)
- Falls back to top recipe if no match found

## Key Design Decisions

### Why LangGraph?

- **State Management**: Built-in state management simplifies passing data between agents
- **Conditional Routing**: Easy to implement complex routing logic
- **Visualization**: Can visualize the agent graph for debugging
- **Checkpointing**: Built-in support for conversation persistence (though we use PostgreSQL)

### Why Separate Agents?

- **Single Responsibility**: Each agent does one thing well
- **Easier Testing**: Can test each agent in isolation
- **Flexible Routing**: Can skip agents when not needed (e.g., skip search for comparisons)
- **Clear Debugging**: Can see exactly which agent processed what

### Why Pinecone for Vector Search?

- **Semantic Understanding**: Finds recipes based on meaning, not just keywords
- **Scalability**: Handles large recipe datasets efficiently
- **Metadata Filtering**: Can combine vector search with metadata filters (future enhancement)
- **Managed Service**: Reduces infrastructure overhead

### Why PostgreSQL for Conversation History?

- **Relational Structure**: Natural fit for conversation data
- **ACID Compliance**: Ensures data integrity
- **Query Flexibility**: Easy to query by conversation_id, timestamp, etc.
- **Mature Ecosystem**: Well-supported with SQLAlchemy and Alembic

## Future Enhancements

While the current architecture is functional, potential improvements include:

1. **Image Generation Agent**: Add DALL-E or Stable Diffusion integration for recipe images
2. **Caching Layer**: Cache embeddings and search results to reduce API calls
3. **Recipe Filtering**: Add metadata filters in Pinecone (e.g., filter by cuisine before vector search)
4. **User Profiles**: Store user preferences across sessions
5. **Recipe Ratings**: Allow users to rate recipes and improve recommendations
6. **Batch Processing**: Process multiple recipe requests in parallel

## Conclusion

TastyAI demonstrates a practical application of agentic architecture for a real-world problem. By breaking down recipe recommendation into specialized agents, the system achieves:

- **Modularity**: Each component has a clear purpose
- **Maintainability**: Easy to understand and modify
- **Scalability**: Can improve individual agents without affecting others
- **User Experience**: Natural, context-aware conversations

The architecture balances complexity with practicality, using modern tools (LangGraph, OpenAI, Pinecone) while maintaining code clarity and developer experience.
