from langgraph.graph import StateGraph, END
from mcp.schema import TastyAIState, Message
from agents.parser_agent import parse_user_input
from agents.search_agent import search_recipes
from agents.response_agent import generate_response
from agents.conversation_agent import classify_intent
from models.schema import UserPreferences
from db.services import save_message_to_db
import json
import uuid

def intent_node(state: TastyAIState) -> dict:
    # Convert Message objects to dicts for classify_intent
    messages_list = state.messages or []
    messages_dicts = [
        m.dict() if hasattr(m, 'dict') else (
            m.model_dump() if hasattr(m, 'model_dump') else m
        )
        for m in messages_list
    ]
    
    intent = classify_intent.invoke({"messages": messages_dicts})
    return {
        **state.dict(),
        "intent": intent.get("intent")
    }

def parser_node(state: TastyAIState) -> dict:
    new_user_message = Message(role="user", content=state.user_input)

    conversation_id = state.conversation_id or str(uuid.uuid4())
    save_message_to_db(conversation_id, "user", state.user_input)

    updated_messages = state.messages + [new_user_message]

    preferences = parse_user_input.invoke({"messages": updated_messages})

    # Ensure plain dict (some tools return Pydantic or LangChain objects)
    if hasattr(preferences, "dict"):
        preferences = preferences.dict()
    elif hasattr(preferences, "model_dump"):
        preferences = preferences.model_dump()

    preferences_obj = UserPreferences(**preferences) if isinstance(preferences, dict) else preferences

    return {
        **state.dict(),
        "messages": updated_messages,
        "preferences": preferences_obj,
        "conversation_id": conversation_id
    }

def search_node(state: TastyAIState) -> dict:
    """
    Search for recipes based on user preferences.
    
    OPTIMIZATION: Skip search if this is a follow-up question and we already have results.
    This prevents:
    - Redundant API calls to Pinecone
    - Context loss (getting different recipes than what user was discussing)
    - Slower response times
    """

    print("######################################################")
    print("SEARCHING NOW...")
    print("######################################################")
    
    # Check if this is a follow-up question
    # is_followup = is_follow_up_question(state.messages)
    
    # # If it's a follow-up and we have existing results, reuse them
    # if is_followup and state.results:
    #     print("********************************************************")
    #     print("🔄 FOLLOW-UP QUESTION DETECTED - REUSING EXISTING SEARCH RESULTS")
    #     print(f"📋 PRESERVING {len(state.results)} RECIPES FROM PREVIOUS SEARCH")
    #     return {**state.dict()}
    
    # if not state.results:
    #     print("********************************************************")
    #     print("🔍 NEW SEARCH REQUEST DETECTED - QUERYING DATABASE")
    
    # Convert preferences to dict for search_recipes
    prefs_dict = state.preferences.dict() if hasattr(state.preferences, 'dict') else (
        state.preferences.model_dump() if hasattr(state.preferences, 'model_dump') else state.preferences
    )
    # print("SEARCHING RESULT: prefs_dict", prefs_dict)
    results = search_recipes.invoke({"preferences": prefs_dict})
    # print("SEARCHING RESULT: results", results)
    
    return {
      **state.dict(),
      "results": results["matches"]
    }

def response_node(state: TastyAIState) -> dict:
    # If routing directly to response (skipping parser), ensure user message is added and saved
    updated_messages = state.messages
    if state.user_input:
        # Check if the last message is already the user input (might already be added in main.py)
        last_msg = state.messages[-1] if state.messages else None
        if not last_msg or (hasattr(last_msg, 'content') and last_msg.content != state.user_input):
            new_user_message = Message(role="user", content=state.user_input)
            updated_messages = state.messages + [new_user_message]
            # Save to DB
            conversation_id = state.conversation_id or str(uuid.uuid4())
            save_message_to_db(conversation_id, "user", state.user_input)
    
    # Convert preferences to dict for generate_response
    # Handle case where preferences might be None (e.g., when routing directly to response)
    updated_preferences = state.preferences
    if state.preferences is None:
        # For follow-up questions, create a minimal preferences dict
        # Try to extract language from previous preferences in messages if available
        language = "English"  # default
        prefs_dict = {
            "language": language,
            "cuisine": "unknown",
            "diet": "unknown",
            "dish": "unknown",
            "ingredients": [],
            "allergies": [],
            "meal_type": "unknown",
            "cooking_time": "unknown"
        }
        # Convert to UserPreferences object for state
        updated_preferences = UserPreferences(**prefs_dict)
    else:
        prefs_dict = state.preferences.dict() if hasattr(state.preferences, 'dict') else (
            state.preferences.model_dump() if hasattr(state.preferences, 'model_dump') else state.preferences
        )
    
    # Convert RecipeMatch objects to dicts for generate_response
    results_list = state.results or []
    results_dicts = [
        r.dict() if hasattr(r, 'dict') else (
            r.model_dump() if hasattr(r, 'model_dump') else r
        )
        for r in results_list
    ]
    
    # Convert Message objects to dicts for generate_response
    messages_list = updated_messages or []
    messages_dicts = [
        m.dict() if hasattr(m, 'dict') else (
            m.model_dump() if hasattr(m, 'model_dump') else m
        )
        for m in messages_list
    ]
    
    # print("RESPONSE RESULT: results_dicts", results_dicts)
    result = generate_response.invoke({
        "preferences": prefs_dict,
        "results": results_dicts,
        "messages": messages_dicts,
        "intent": state.intent
    })

    assistant_msg = result.get("generated_response")
    if assistant_msg:
        save_message_to_db(state.conversation_id, "assistant", assistant_msg)

    # print("RESPONSE RESULT: result", result)

    return {
        **state.dict(),
        "messages": updated_messages,
        "preferences": updated_preferences,
        "generated_response": result.get("generated_response") if isinstance(result, dict) else None
    }

def route_after_intent(state: TastyAIState) -> str:
    """
    Route based on intent classification.
    
    Returns the intent value which maps to a target node in the conditional edges.
    The mapping is:
        - "new_search" -> "parse" (need to parse preferences and search)
        - "comparison" -> "response" (reuse existing results)
        - "recipe_request" -> "response" (reuse existing results)
        - "general" -> "response" (reuse existing results)
    """
    intent = getattr(state, 'intent', 'new_search')
    
    print(f"🔀 Routing decision: intent={intent}")
    
    if intent == "new_search":
        print("   → GOING TO PARSE (new query)")
        return "new_search"
    elif intent == "comparison":
        print("   → GOING TO RESPONSE (follow-up, reusing results)")
        return "comparison"
    elif intent == "recipe_request":
        print("   → GOING TO RESPONSE (follow-up, reusing results)")
        return "recipe_request"
    elif intent == "general":
        print("   → GOING TO RESPONSE (follow-up, reusing results)")
        return "general"
    else:
        print("   → UNKNOWN INTENT (defaulting to new_search)")
        return "new_search"

# Build LangGraph flow
def build_graph():
    builder = StateGraph(TastyAIState)

    # Add parser agent node
    builder.add_node("intent", intent_node)
    builder.add_node("parse", parser_node)
    builder.add_node("search", search_node)
    builder.add_node("response", response_node)

    builder.set_entry_point("intent")

    # Add conditional routing after intent classification
    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "new_search": "parse",
            "comparison": "response",
            "recipe_request": "response",
            "general": "response"
        }
    )

    # Set entry point and end
    builder.add_edge("parse", "search")
    builder.add_edge("search", "response")
    builder.add_edge("response", END)

    return builder.compile()
