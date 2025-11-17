# agents/response_agent.py

from langchain_core.tools import tool
from typing import Dict, List
from .services import (
    handle_comparative_response,
    handle_general_response,
    handle_recipe_request,
    handle_new_search
)

@tool
def generate_response(preferences: Dict, results: List[Dict], messages: List[Dict] = [], intent: str = "new_search") -> Dict:
    """
    Generate a conversational, multilingual meal recommendation based on
    user preferences, search results, and summarized chat history.
    """

    language = preferences.get("language", "English")

    if intent == "comparison":
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return handle_comparative_response(results, last_user_msg, language, messages)
    elif intent == "general":
        return handle_general_response(preferences, results, messages, language)
    elif intent == "recipe_request":
        return handle_recipe_request(preferences, results, messages, language)
    else:
        return handle_new_search(preferences, results, messages, language)
