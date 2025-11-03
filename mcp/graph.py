from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from mcp.schema import TastyAIState, Message
from agents.parser_agent import parse_user_input
from agents.search_agent import search_recipes
from agents.response_agent import generate_response
from agents.decide_recipe_agent import decide_recipe_request
from agents.format_agent import format_recipe
from models.schema import UserPreferences
from db.services import save_message_to_db
import json
import uuid

def parser_node(state: TastyAIState) -> dict:
    print("#####################################################################")
    print("PARSER NODE: state", state)
    print("#####################################################################")
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

    # Convert to UserPreferences object for state validation
    preferences_obj = UserPreferences(**preferences) if isinstance(preferences, dict) else preferences

    # print("PARSING RESULT: preferences_obj", preferences_obj)

    return {
        **state.dict(),
        "messages": updated_messages,
        "preferences": preferences_obj,
        "conversation_id": conversation_id
    }

def search_node(state: TastyAIState) -> dict:
    print("#####################################################################")
    print("SEARCH NODE: state", state)
    print("#####################################################################")
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
    print("#####################################################################")
    print("RESPONSE NODE: state", state)
    print("#####################################################################")
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
    # print("RESPONSE RESULT: results_dicts", results_dicts)
    result = generate_response.invoke({
        "preferences": prefs_dict,
        "results": results_dicts
    })

    assistant_msg = result.get("generated_response")
    if assistant_msg:
        save_message_to_db(state.conversation_id, "assistant", assistant_msg)

    # print("RESPONSE RESULT: result", result)

    full_history = "\n".join([
        f"{m.role.capitalize()}: {m.content}" for m in state.messages + [Message(role="assistant", content=assistant_msg)]
    ])

    return {
        **state.dict(),
        "generated_response": result.get("generated_response") if isinstance(result, dict) else None,
        "full_history": full_history
    }

def decide_recipe_node(state: TastyAIState) -> dict:
    print("#####################################################################")
    print("DECIDE RECIPE NODE: state", state)
    print("#####################################################################")

    result = decide_recipe_request.invoke({
        "input": {"message_history": state.full_history}
    })

    # ✅ Ensure we store plain dict, not Pydantic model
    if hasattr(result, "dict"):
        result = result.dict()
    elif hasattr(result, "model_dump"):
        result = result.model_dump()

    return {
        **state.dict(),
        "decide_recipe_request": result
    }

def format_recipe_node(state: TastyAIState) -> dict:
    print("#####################################################################")
    print("FORMAT RECIPE NODE: state", state)
    print("#####################################################################")

    top_recipe = state.results[0] if state.results else {}

    # ✅ Convert to dict if necessary
    if hasattr(top_recipe, "dict"):
        top_recipe = top_recipe.dict()
    elif hasattr(top_recipe, "model_dump"):
        top_recipe = top_recipe.model_dump()

    # ✅ Wrap in { "recipe": ... }
    response = format_recipe.invoke({"recipe": top_recipe})

    if response:
        save_message_to_db(state.conversation_id, "assistant", response)

    return {
        **state.dict(),
        "generated_response": response
    }

# Build LangGraph flow
def build_graph():
    builder = StateGraph(TastyAIState)

    # Add parser agent node
    builder.add_node("parse", parser_node)
    builder.add_node("search", search_node)
    builder.add_node("response", response_node)
    builder.add_node("decide_recipe_request", decide_recipe_node)
    builder.add_node("format_recipe", format_recipe_node)

    # Set entry point and end
    builder.set_entry_point("parse")
    builder.add_edge("parse", "search")
    builder.add_edge("search", "response")
    builder.add_conditional_edges(
        "response",
        lambda state: "decide_recipe_request",
        {
            "decide_recipe_request": "decide_recipe_request"
        }
    )

    builder.add_conditional_edges(
        "decide_recipe_request",
        lambda state: state.decide_recipe_request["user_wants_recipe"],
        {
            "yes": "format_recipe",
            "no": END
        }
    )

    builder.add_edge("format_recipe", END)

    return builder.compile()
