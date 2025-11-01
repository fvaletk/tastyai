from langgraph.graph import StateGraph, END
from mcp.schema import TastyAIState, Message
from agents.parser_agent import parse_user_input
from agents.search_agent import search_recipes
from agents.response_agent import generate_response
from models.schema import UserPreferences
import json

def parser_node(state: TastyAIState) -> dict:
    new_user_message = Message(role="user", content=state.user_input)
    updated_messages = state.messages + [new_user_message]

    preferences = parse_user_input.invoke({"messages": updated_messages})

    # Ensure plain dict (some tools return Pydantic or LangChain objects)
    if hasattr(preferences, "dict"):
        preferences = preferences.dict()
    elif hasattr(preferences, "model_dump"):
        preferences = preferences.model_dump()

    # Convert to UserPreferences object for state validation
    preferences_obj = UserPreferences(**preferences) if isinstance(preferences, dict) else preferences

    return {
        **state.dict(),
        "messages": updated_messages,
        "preferences": preferences_obj
    }

def search_node(state: TastyAIState) -> dict:
    # Convert preferences to dict for search_recipes
    prefs_dict = state.preferences.dict() if hasattr(state.preferences, 'dict') else (
        state.preferences.model_dump() if hasattr(state.preferences, 'model_dump') else state.preferences
    )
    results = search_recipes.invoke({"preferences": prefs_dict})
    return {
      **state.dict(),
      "results": results["matches"]
    }

def response_node(state: TastyAIState) -> dict:
    # Convert preferences to dict for generate_response
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
    
    result = generate_response.invoke({
        "preferences": prefs_dict,
        "results": results_dicts
    })

    return {
        **state.dict(),
        "generated_response": result.get("generated_response") if isinstance(result, dict) else None
    }

# Build LangGraph flow
def build_graph():
    builder = StateGraph(TastyAIState)

    # Add parser agent node
    builder.add_node("parse", parser_node)
    builder.add_node("search", search_node)
    builder.add_node("response", response_node)

    # Set entry point and end
    builder.set_entry_point("parse")
    builder.add_edge("parse", "search")
    builder.add_edge("search", "response")
    builder.add_edge("response", END)

    return builder.compile()
