from langgraph.graph import StateGraph, END
from mcp.schema import TastyAIState, Message
from agents.parser_agent import parse_user_input
from agents.search_agent import search_recipes
from agents.response_agent import generate_response
from agents.conversation_agent import classify_intent
from agents.recipe_request_agent import analyze_recipe_request
from agents.services import handle_new_search
from models.schema import UserPreferences
from db.services import save_message_to_db
import json
import uuid

def intent_node(state: TastyAIState) -> dict:
    # Save user message to database (always runs first, regardless of intent)
    # This ensures all user messages are saved, even for "general" or "comparison" intents
    conversation_id = state.conversation_id
    if state.user_input:
        conversation_id = state.conversation_id or str(uuid.uuid4())
        print("######################################################")
        print("SAVING USER MESSAGE TO DB ON INTENT NODE")
        print("######################################################")
        save_message_to_db(conversation_id, "user", state.user_input)
    
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
        "intent": intent.get("intent"),
        "intent_reasoning": intent.get("reasoning", ""),
        "conversation_id": conversation_id
    }

def parser_node(state: TastyAIState) -> dict:
    new_user_message = Message(role="user", content=state.user_input)

    conversation_id = state.conversation_id or str(uuid.uuid4())
    # Note: User message is already saved in intent_node, but we ensure it's in messages list
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
    # If routing directly to response (skipping parser), ensure user message is in messages list
    # Note: User message is already saved in intent_node
    updated_messages = state.messages
    if state.user_input:
        # Check if the last message is already the user input (might already be added in main.py)
        last_msg = state.messages[-1] if state.messages else None
        if not last_msg or (hasattr(last_msg, 'content') and last_msg.content != state.user_input):
            new_user_message = Message(role="user", content=state.user_input)
            updated_messages = state.messages + [new_user_message]
    
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
    
    # For recipe_request, also include previous_results if available
    # This helps when user is referring to recipes from earlier in the conversation
    if state.intent == "recipe_request" and state.previous_results:
        previous_results_dicts = [
            r.dict() if hasattr(r, 'dict') else (
                r.model_dump() if hasattr(r, 'model_dump') else r
            )
            for r in state.previous_results
        ]
        # Combine current and previous results, prioritizing current
        all_results = results_dicts + [r for r in previous_results_dicts if r not in results_dicts]
        results_dicts = all_results
        print(f"📚 Combined {len(results_dicts)} results (current + previous) for recipe request")
    
    # Check if recipe_request intent has no results - return helpful message
    if state.intent == "recipe_request" and not results_dicts:
        print("⚠️ Recipe request with no results - returning helpful message")
        helpful_message = "I can't find what you're looking for. Please give me more details so I can try again."
        
        save_message_to_db(state.conversation_id, "assistant", helpful_message)
        
        return {
            **state.dict(),
            "messages": updated_messages,
            "preferences": updated_preferences,
            "generated_response": helpful_message
        }
    
    # Convert Message objects to dicts for generate_response
    messages_list = updated_messages or []
    messages_dicts = [
        m.dict() if hasattr(m, 'dict') else (
            m.model_dump() if hasattr(m, 'model_dump') else m
        )
        for m in messages_list
    ]
    
    # Pass matched_recipe_title if available (from recipe_request_analysis_node)
    matched_recipe_title = getattr(state, 'matched_recipe_title', None)
    
    # print("RESPONSE RESULT: results_dicts", results_dicts)
    result = generate_response.invoke({
        "preferences": prefs_dict,
        "results": results_dicts,
        "messages": messages_dicts,
        "intent": state.intent,
        "matched_recipe_title": matched_recipe_title
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

def recipe_request_analysis_node(state: TastyAIState) -> dict:
    """
    Analyzes recipe requests to determine if user wants:
    - A specific recipe (go to response_node)
    - A dish that was shown before (show recipes and END)
    - A new dish (go to parse/search)
    """
    # Convert messages and results to dicts
    messages_list = state.messages or []
    messages_dicts = [
        m.dict() if hasattr(m, 'dict') else (
            m.model_dump() if hasattr(m, 'model_dump') else m
        )
        for m in messages_list
    ]
    
    results_list = state.results or []
    results_dicts = [
        r.dict() if hasattr(r, 'dict') else (
            r.model_dump() if hasattr(r, 'model_dump') else r
        )
        for r in results_list
    ]
    
    # Also include previous_results if available
    if state.previous_results:
        previous_results_dicts = [
            r.dict() if hasattr(r, 'dict') else (
                r.model_dump() if hasattr(r, 'model_dump') else r
            )
            for r in state.previous_results
        ]
        all_results = results_dicts + [r for r in previous_results_dicts if r not in results_dicts]
        results_dicts = all_results
    
    # Pass intent reasoning to help with analysis
    intent_reasoning = getattr(state, 'intent_reasoning', '') or ""
    
    analysis = analyze_recipe_request.invoke({
        "messages": messages_dicts,
        "results": results_dicts,
        "intent_reasoning": intent_reasoning
    })
    
    return {
        **state.dict(),
        "recipe_request_type": analysis.get("type"),
        "matched_recipe_title": analysis.get("matched_recipe_title"),
        "dish_name": analysis.get("dish_name")
    }

def show_recipes_node(state: TastyAIState) -> dict:
    """
    Shows recipes for a dish that was discussed before.
    Uses handle_new_search to format the recipes nicely, then ENDs.
    """
    # Convert preferences to dict
    prefs_dict = {}
    updated_preferences = state.preferences
    if state.preferences:
        prefs_dict = state.preferences.dict() if hasattr(state.preferences, 'dict') else (
            state.preferences.model_dump() if hasattr(state.preferences, 'model_dump') else state.preferences
        )
    else:
        prefs_dict = {
            "language": "English",
            "cuisine": "unknown",
            "diet": "unknown",
            "dish": state.dish_name or "unknown",
            "ingredients": [],
            "allergies": [],
            "meal_type": "unknown",
            "cooking_time": "unknown"
        }
        # Create UserPreferences object for state
        updated_preferences = UserPreferences(**prefs_dict)
    
    # Convert results to dicts
    results_list = state.results or []
    results_dicts = [
        r.dict() if hasattr(r, 'dict') else (
            r.model_dump() if hasattr(r, 'model_dump') else r
        )
        for r in results_list
    ]
    
    # Also include previous_results if available
    if state.previous_results:
        previous_results_dicts = [
            r.dict() if hasattr(r, 'dict') else (
                r.model_dump() if hasattr(r, 'model_dump') else r
            )
            for r in state.previous_results
        ]
        all_results = results_dicts + [r for r in previous_results_dicts if r not in results_dicts]
        results_dicts = all_results
    
    # Filter results by dish if dish_name is specified
    if state.dish_name:
        dish_lower = state.dish_name.lower()
        filtered_results = []
        for recipe in results_dicts:
            recipe_title = recipe.get("title", "").lower()
            # Include recipe if dish name is in the title
            if dish_lower in recipe_title:
                filtered_results.append(recipe)
        
        if filtered_results:
            results_dicts = filtered_results
            print(f"🍽️ Filtered to {len(results_dicts)} recipes matching dish: {state.dish_name}")
    
    # Convert messages to dicts
    messages_list = state.messages or []
    messages_dicts = [
        m.dict() if hasattr(m, 'dict') else (
            m.model_dump() if hasattr(m, 'model_dump') else m
        )
        for m in messages_list
    ]
    
    # Use handle_new_search to format recipes nicely
    result = handle_new_search(prefs_dict, results_dicts, messages_dicts, prefs_dict.get("language", "English"))
    
    assistant_msg = result.get("generated_response")
    if assistant_msg:
        save_message_to_db(state.conversation_id, "assistant", assistant_msg)
    
    return {
        **state.dict(),
        "preferences": updated_preferences,
        "generated_response": assistant_msg
    }

def route_after_recipe_analysis(state: TastyAIState) -> str:
    """
    Route after recipe request analysis.
    
    Returns:
        - "response" if specific_recipe (user wants a specific recipe)
        - "show_recipes" if dish (user wants to see dish recipes again - will END)
        - "parse" if new_dish (user wants a new dish - needs search)
    """
    request_type = getattr(state, 'recipe_request_type', 'new_dish')
    
    print(f"🔀 Recipe request routing: type={request_type}")
    
    if request_type == "specific_recipe":
        print("   → GOING TO RESPONSE (specific recipe requested)")
        return "response"
    elif request_type == "dish":
        print("   → SHOWING RECIPES (dish was shown before)")
        return "show_recipes"
    else:  # new_dish
        print("   → GOING TO PARSE (new dish, needs search)")
        return "parse"

def route_after_intent(state: TastyAIState) -> str:
    """
    Route based on intent classification.
    
    Returns the intent value which maps to a target node in the conditional edges.
    The mapping is:
        - "new_search" -> "parse" (need to parse preferences and search)
        - "comparison" -> "response" (reuse existing results)
        - "recipe_request" -> "recipe_analysis" (analyze the request)
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
        print("   → GOING TO RECIPE ANALYSIS (analyzing request)")
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

    # Add nodes
    builder.add_node("intent", intent_node)
    builder.add_node("parse", parser_node)
    builder.add_node("search", search_node)
    builder.add_node("recipe_analysis", recipe_request_analysis_node)
    builder.add_node("show_recipes", show_recipes_node)
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

    # Add conditional routing after recipe analysis
    builder.add_conditional_edges(
        "recipe_analysis",
        route_after_recipe_analysis,
        {
            "response": "response",
            "show_recipes": "show_recipes",
            "parse": "parse"
        }
    )

    # Set edges
    builder.add_edge("parse", "search")
    builder.add_edge("search", "response")
    builder.add_edge("response", END)
    builder.add_edge("show_recipes", END)  # END after showing recipes

    return builder.compile()
