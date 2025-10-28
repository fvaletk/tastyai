from langgraph.graph import StateGraph, END
from mcp.schema import TastyAIState
from agents.parser_agent import parse_user_input
from agents.search_agent import search_recipes
from agents.response_agent import generate_response
import json

def parser_node(state: TastyAIState) -> dict:
    preferences = parse_user_input.invoke(state.user_input)
    return {
        "user_input": state.user_input,
        "preferences": preferences
    }

def search_node(state: TastyAIState) -> dict:
    results = search_recipes.invoke({"preferences": state.preferences})
    return {
      **state.dict(),
      "results": results["matches"]
    }

def response_node(state: TastyAIState) -> dict:
    result = generate_response.invoke({
        "preferences": state.preferences,
        "results": state.results or []
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
