# agents/recipe_request_agent.py

import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Dict, List
import json
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def analyze_recipe_request(messages: List[Dict], results: List[Dict] = None, intent_reasoning: str = "") -> Dict:
    """
    Analyzes if the user is requesting:
    1. A specific recipe that was mentioned before (return "specific_recipe")
    2. A dish that was discussed before and recipes were shown (return "dish" - show recipes again)
    3. A new dish (return "new_dish" - needs search)
    
    Args:
        messages: Conversation history
        results: Current recipe results
        intent_reasoning: Reasoning from intent classification (from conversation_agent)
    
    Returns:
    {
        "type": "specific_recipe" | "dish" | "new_dish",
        "matched_recipe_title": "title if specific_recipe",
        "dish_name": "dish name if dish",
        "reasoning": "explanation"
    }
    """
    
    if not messages:
        return {"type": "new_dish", "reasoning": "No conversation history"}
    
    # Get last user message
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    
    if not last_user_msg:
        return {"type": "new_dish", "reasoning": "No user message found"}
    
    # Extract recipe titles from previous assistant messages
    # Process messages in reverse order (most recent first) to prioritize recent recipes
    previously_shown_recipes = []
    seen_titles = set()  # Track seen titles (case-insensitive) to avoid duplicates
    excluded_words = {'crust', 'ingredients', 'toppings', 'directions', 'source', 'recipe', 'option', 'choice'}
    
    if messages:
        # Process messages in reverse to get most recent recipes first
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # First, try to extract from numbered lists (most reliable for ordering)
                numbered_pattern = r'(\d+)\.\s+(?:[\*\*]?)([^\n\*\:]+?)(?:[\*\*]?)(?:\s*—|\s*:|\n|$)'
                numbered_matches = re.findall(numbered_pattern, content)
                for num_str, title in numbered_matches:
                    title = title.strip().strip('*').strip()
                    title_lower = title.lower()
                    # Check if it's a valid recipe title (not excluded word, has reasonable length)
                    if (title and len(title) > 3 and 
                        title_lower not in excluded_words and
                        not all(word in excluded_words for word in title_lower.split())):
                        if title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            previously_shown_recipes.append(title)
                
                # Then extract from bolded titles **Title**
                bold_pattern = r'\*\*([^*\n]+?)\*\*'
                bold_matches = re.findall(bold_pattern, content)
                for title in bold_matches:
                    title = title.strip()
                    title_lower = title.lower()
                    if (title and len(title) > 3 and 
                        title_lower not in excluded_words and
                        not all(word in excluded_words for word in title_lower.split())):
                        if title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            previously_shown_recipes.append(title)
                
                # Extract from markdown headers ### 🍽️ Title or ## 🍽️ Title
                header_pattern = r'#+\s+🍽️\s+([^\n]+)'
                header_matches = re.findall(header_pattern, content)
                for title in header_matches:
                    title = title.strip()
                    title_lower = title.lower()
                    if (title and len(title) > 3 and 
                        title_lower not in excluded_words and
                        not all(word in excluded_words for word in title_lower.split())):
                        if title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            previously_shown_recipes.append(title)
    
    # Reverse back to chronological order (oldest first) for ordinal mapping
    # This way "the first one" = previously_shown_recipes[0], "the second one" = previously_shown_recipes[1]
    previously_shown_recipes = list(reversed(previously_shown_recipes))
    
    print(f"📋 Previously shown recipes: {previously_shown_recipes}")
    print(f"🔍 User request: {last_user_msg}")
    if intent_reasoning:
        print(f"💭 Intent reasoning: {intent_reasoning}")
    
    # Analyze with LLM using intent reasoning as context
    context = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')[:200]}" for m in messages[-6:]])
    
    prompt = f"""
    Analyze the user's latest message to determine what they're requesting.

    Intent Classification Reasoning (from conversation analysis):
    {intent_reasoning if intent_reasoning else "No specific reasoning provided"}

    Conversation context (last 6 messages):
    {context}

    User's latest message: "{last_user_msg}"

    Previously shown recipes in this conversation (in order of appearance):
    {json.dumps([f"{i+1}. {recipe}" for i, recipe in enumerate(previously_shown_recipes)], indent=2) if previously_shown_recipes else "None"}
    
    IMPORTANT: When user says "the first one/option", they mean recipe #1 in the list above.
    When user says "the second one/option", they mean recipe #2 in the list above, etc.

    Available results (current recipes):
    {json.dumps([r.get('title', '') for r in (results or [])[:5]], indent=2) if results else "None"}

    Determine if the user is requesting:
    1. "specific_recipe" - User wants a specific recipe that was mentioned/shown before
       - User mentions a specific recipe name from the previously shown recipes
       - User says "the first one", "the second one", "the third one", "that one", "this one", "the last one"
       - User says "give me the recipe for [specific name]"
       - User is selecting one of the recipes that were already presented
       - IMPORTANT: If user says "the second option" or "the second one", map it to the SECOND recipe in the previously_shown_recipes list
       - IMPORTANT: If user says "the first option" or "the first one", map it to the FIRST recipe in the previously_shown_recipes list
       - Examples: "I want the Japanese Pie recipe", "Show me the first one", "Give me the recipe for Classic Italian Pizza", "Give me the recipe for the second option"

    2. "dish" - User is talking about a dish type, and recipes for this dish were already shown
       - User mentions a dish type/category but NOT a specific recipe name
       - Recipes for this dish type were shown in previous messages
       - User wants to see those recipes again or is still deciding between them
       - User is changing their mind back to a dish that was discussed earlier
       - Examples: "I want pies" (when pie recipes were shown), "Actually I want pizza" (when pizza recipes were shown), "Never mind I decided I want to go with pies"

    3. "new_dish" - User is requesting a completely new dish that wasn't discussed
       - No recipes for this dish were shown before
       - This is a new search request that needs to go through parse/search
       - Examples: "I want pasta" (when no pasta recipes were shown), "Show me dessert recipes"

    Use the intent reasoning to help understand the user's intent. If the reasoning suggests the user is referring to something already shown, 
    check if it's a specific recipe or a dish type.
    
    CRITICAL FOR ORDINAL REFERENCES:
    - If user says "the first one/option/recipe", matched_recipe_title should be the recipe at index 0 in the numbered list above
    - If user says "the second one/option/recipe", matched_recipe_title should be the recipe at index 1 in the numbered list above
    - If user says "the third one/option/recipe", matched_recipe_title should be the recipe at index 2 in the numbered list above
    - Always use the EXACT title from the numbered list above, preserving capitalization and punctuation
    - If the user's request refers to a comparison (e.g., "the second option" after seeing a comparison), use the order from the most recent comparison message

    Respond with ONLY a JSON object:
    {{
      "type": "specific_recipe" | "dish" | "new_dish",
      "matched_recipe_title": "exact recipe title if specific_recipe, else null",
      "dish_name": "dish name if dish, else null",
      "reasoning": "brief explanation"
    }}

    CRITICAL: Your entire response must be valid JSON only. No other text.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing recipe requests. Respond only with valid JSON."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2
        )
        
        result = response.choices[0].message.content.strip()
        result = result.replace("```json\n", "").replace("```\n", "").replace("```", "").strip()
        
        parsed = json.loads(result)
        
        print("######################################################")
        print(f"🍽️ RECIPE REQUEST ANALYSIS: {parsed.get('type')}")
        print(f"💭 REASONING: {parsed.get('reasoning')}")
        if parsed.get('matched_recipe_title'):
            print(f"📝 Matched recipe: {parsed.get('matched_recipe_title')}")
        if parsed.get('dish_name'):
            print(f"🍕 Dish: {parsed.get('dish_name')}")
        print("######################################################")
        
        return parsed
        
    except Exception as e:
        print(f"⚠️ Recipe request analysis failed: {e}")
        import traceback
        print(f"⚠️ Traceback: {traceback.format_exc()}")
        
        # Simple fallback: if recipes were shown and user message is short, assume dish
        # Otherwise assume new_dish
        if previously_shown_recipes and len(last_user_msg.split()) <= 5:
            return {
                "type": "dish",
                "matched_recipe_title": None,
                "dish_name": None,
                "reasoning": f"Fallback: short message with previously shown recipes"
            }
        
        return {
            "type": "new_dish",
            "matched_recipe_title": None,
            "dish_name": None,
            "reasoning": "Fallback: new dish"
        }
