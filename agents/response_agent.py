# agents/response_agent.py

from langchain_core.tools import tool
from typing import Dict, List, Optional
from .services import (
    handle_comparative_response,
    handle_general_response,
    handle_recipe_request,
    handle_new_search
)

@tool
def generate_response(preferences: Dict, results: List[Dict], messages: List[Dict] = [], intent: str = "new_search", matched_recipe_title: Optional[str] = None) -> Dict:
    """
    Generate a conversational, multilingual meal recommendation based on
    user preferences, search results, and summarized chat history.
    
    Args:
        matched_recipe_title: The exact recipe title identified by recipe_request_analysis_node.
                            Used when intent is "recipe_request" to find the specific recipe.
    """

    language = preferences.get("language", "English")

    if intent == "comparison":
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return handle_comparative_response(results, last_user_msg, language, messages)
    elif intent == "general":
        return handle_general_response(preferences, results, messages, language)
    elif intent == "recipe_request":
        return handle_recipe_request(preferences, results, messages, language, matched_recipe_title)
    else:
        return handle_new_search(preferences, results, messages, language)
