# agents/services.py

import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
import json
from .helpers import format_recipe, summarize_conversation

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def handle_comparative_response(results: List[Dict], user_msg: str, language: str, messages: List[Dict] = None) -> Dict:
    """
    Responds to user follow-up questions comparing or refining recipe suggestions.
    Enhanced to provide actual data from recipe directions for accurate time comparisons.
    """

    print("######################################################")
    print("COMPARATIVE RESPONSE")
    print("######################################################")
    # Extract actual recipe data for comparison
    recipe_details = []
    for idx, recipe in enumerate(results[:3]):
        title = recipe.get("title", f"Recipe {idx+1}")
        ingredients_count = len(recipe.get("ingredients", []))
        directions_count = len(recipe.get("directions", []))
        directions = recipe.get("directions", [])
        
        recipe_details.append({
            "title": title,
            "ingredients_count": ingredients_count,
            "steps_count": directions_count,
            "directions": directions
        })

    prompt = f"""
    The user previously received the following recipe suggestions:

    {json.dumps(recipe_details, indent=2)}

    Full conversation history:
    {json.dumps(messages[-6:] if messages else [], indent=2)}

    The user's latest question: "{user_msg}"

    Respond in {language}, as a friendly home cook helping someone decide between options.
    
    IMPORTANT INSTRUCTIONS:
    - Look at the actual recipe directions to estimate preparation time
    - Count the number of steps and complexity of each step
    - Recipes with fewer steps and simpler instructions generally take less time
    - Be specific when comparing - use the actual step counts and ingredient counts
    - Focus on what the user asked about (prep time, ingredients, healthiness, etc.)
    - Keep it brief, warm, and helpful
    - Do NOT list all recipes again unless directly relevant
    - Answer their specific question directly first, then add context if helpful
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a warm and concise food assistant who provides accurate comparisons based on recipe data."},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.7,
        )
        return {"generated_response": response.choices[0].message.content.strip()}
    except Exception as e:
        print("❌ Comparison response failed:", e)
        return {"generated_response": "Sorry, I couldn't compare those recipes right now. Try rephrasing?"}

def handle_general_response(preferences: Dict, results: List[Dict], messages: List[Dict] = None, language: str = "English") -> Dict:
    """
    Responds to general questions about recipes such as:
    - Carbohydrates/nutritional information
    - Preparation/cooking time
    - Health considerations (diabetes, allergies, etc.)
    - Ingredient information
    """

    print("######################################################")
    print("GENERAL RESPONSE")
    print("######################################################")

    if not results:
        return {"generated_response": "Sorry, I couldn't find any recipes to answer your question about."}
    
    # Get the last user message to understand what they're asking
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "") if messages else ""
    
    # Get conversation summary for context
    conversation_summary = summarize_conversation(messages) if messages else ""
    
    # Prepare recipe data for analysis
    recipe_data = []
    for recipe in results[:3]:  # Focus on top 3 recipes
        recipe_data.append({
            "title": recipe.get("title", ""),
            "ingredients": recipe.get("ingredients", []),
            "directions": recipe.get("directions", []),
            "source": recipe.get("source", "")
        })
    
    system_prompt = f"""
    You are TastyAI — a multilingual, friendly, food-loving AI chef with expertise in nutrition and cooking.
    
    Always respond in this language: {language}.
    Sound like a knowledgeable home cook or nutritionist talking naturally to a friend.
    Be warm, helpful, and accurate.
    
    The user is asking a general question about the recipes (nutrition, time, health, allergies, etc.).
    Answer their specific question based on the recipe data provided.
    """
    
    user_prompt = f"""
    Conversation context:
    {conversation_summary}
    
    Available recipes:
    {json.dumps(recipe_data, indent=2)}
    
    User's question: "{last_user_msg}"
    
    Based on the recipe information above, answer the user's question. Focus on:
    - Nutritional information (carbs, protein, calories if you can estimate from ingredients)
    - Preparation and cooking time (estimate from the number of steps and complexity)
    - Health considerations (diabetes-friendly, allergy concerns, dietary restrictions)
    - Ingredient details and substitutions
    
    If the question is about a specific recipe, identify which one and answer accordingly.
    If the question is general, provide information about the available recipes.
    Be specific and helpful, but acknowledge when you're making estimates.
    If the user is ultimately been polite and saying stuff like "thank you" or "you're amazing", then just say "You're welcome! Ping me if you need another meal recommendation." and end the conversation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.7,
        )
        return {"generated_response": response.choices[0].message.content.strip()}
    except Exception as e:
        print("❌ General response failed:", e)
        return {"generated_response": "Sorry, I couldn't answer that question right now. Try rephrasing?"}

def handle_recipe_request(preferences: Dict, results: List[Dict], messages: List[Dict] = None, language: str = "English", matched_recipe_title: Optional[str] = None) -> Dict:
    """
    Handles explicit recipe requests. Finds the specific recipe the user wants
    and returns it with full ingredients and directions formatted.
    
    Args:
        matched_recipe_title: The exact recipe title identified by recipe_request_analysis_node.
                            If provided, this takes priority over message-based matching.
    """

    print("######################################################")
    print("RECIPE REQUEST")
    print("######################################################")
    if matched_recipe_title:
        print(f"🎯 Using matched recipe title: {matched_recipe_title}")

    if not results:
        return {"generated_response": "Sorry, I couldn't find any recipes right now."}
    
    # Initialize with top recipe if available
    top_recipe: Optional[Dict] = None
    
    # PRIORITY 1: Use matched_recipe_title if provided (from recipe_request_analysis_node)
    if matched_recipe_title:
        matched_title_lower = matched_recipe_title.lower().strip()
        for recipe in results:
            recipe_title = recipe.get("title", "").strip()
            recipe_title_lower = recipe_title.lower().strip()
            
            # Try exact match (case-insensitive)
            if recipe_title_lower == matched_title_lower:
                top_recipe = recipe
                print(f"✅ Exact match found: {recipe_title}")
                break
            
            # Try partial match - check if matched title is contained in recipe title or vice versa
            if matched_title_lower in recipe_title_lower or recipe_title_lower in matched_title_lower:
                top_recipe = recipe
                print(f"✅ Partial match found: {recipe_title}")
                break
            
            # Try fuzzy match - check if key words match
            matched_words = [w for w in matched_title_lower.split() if len(w) > 3]
            recipe_words = [w for w in recipe_title_lower.split() if len(w) > 3]
            if matched_words and recipe_words:
                # Count how many significant words from matched title appear in recipe title
                matched_count = sum(1 for word in matched_words if word in recipe_title_lower)
                if matched_count >= min(2, len(matched_words)):
                    top_recipe = recipe
                    print(f"✅ Fuzzy match found: {recipe_title} (matched {matched_count} words)")
                    break
    
    # PRIORITY 2: Try to find the specific recipe from user message (fallback)
    if not top_recipe and messages:
        last_user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break
        
        # Check if user mentioned a specific recipe name
        if last_user_msg and results:
            for recipe in results:
                recipe_title = recipe.get("title", "").lower().strip()
                if not recipe_title:
                    continue
                
                # Normalize both strings for comparison
                normalized_recipe = " ".join(recipe_title.split())
                normalized_msg = " ".join(last_user_msg.split())
                
                # Try exact match first
                if normalized_recipe in normalized_msg:
                    top_recipe = recipe
                    print(f"✅ Message-based match found: {recipe_title}")
                    break
                
                # Try matching key words from the recipe title
                recipe_words = [w for w in normalized_recipe.split() if len(w) > 3]
                if len(recipe_words) >= 2:
                    matched_words = sum(1 for word in recipe_words if word in normalized_msg)
                    if matched_words >= 2:
                        top_recipe = recipe
                        print(f"✅ Message-based fuzzy match found: {recipe_title}")
                        break
    
    # PRIORITY 3: Fallback to top recipe if no match found
    if not top_recipe and results:
        top_recipe = results[0]
        print(f"⚠️ No specific match found, using top recipe: {top_recipe.get('title', 'Unknown')}")
    
    # Format and return full recipe with ingredients and directions
    if top_recipe:
        formatted_recipe = format_recipe(top_recipe)
        intro = f"Here's the full recipe for **{top_recipe.get('title', '').strip()}**.\n"
        return {"generated_response": intro + formatted_recipe}
    
    return {"generated_response": "Sorry, I couldn't find that recipe right now. Try rephrasing?"}

def handle_new_search(preferences: Dict, results: List[Dict], messages: List[Dict] = None, language: str = "English") -> Dict:
    """
    Handles new search requests. Shows recipe options without full details,
    inviting the user to select one for the full recipe.
    """
    print("######################################################")
    print("NEW SEARCH")
    print("######################################################")

    language = preferences.get("language", language)
    conversation_summary = summarize_conversation(messages) if messages else ""
    
    # 🌍 Multilingual + conversational configuration
    system_prompt = f"""
    You are TastyAI — a multilingual, friendly, food-loving AI chef.

    Always respond in this language: {language}.
    Sound like a passionate home cook or restaurant chef talking naturally to a friend.
    Avoid repetitive greetings and generic intros; continue the flow of the chat.

    - If the user asks for the recipe, include the full ingredients and directions clearly formatted.
    - If not, discuss and compare recipes naturally.
    - Reference the conversation summary to maintain context.
    - Never use bullet points unless listing ingredients or steps.
    - Be warm, playful, and concise — don't overexplain.
    """

    prefs_text = (
        f"- Cuisine: {preferences.get('cuisine', 'unknown')}\n"
        f"- Dish: {preferences.get('dish', 'unspecified')}\n"
        f"- Ingredients: {', '.join(preferences.get('ingredients', [])) or 'none'}\n"
        f"- Meal type: {preferences.get('meal_type', 'unknown')}\n"
        f"- Cooking time: {preferences.get('cooking_time', 'not specified')}\n"
    )

    # Show recipe options WITHOUT full details (user hasn't selected one yet)
    if results:
        recipe_summaries = "\n\n".join([
            f"{i+1}. **{r['title']}** — made with {', '.join(r['ingredients'][:3]) if r.get('ingredients') else 'various ingredients'}..."
            for i, r in enumerate(results[:5])
        ])

        user_prompt = f"""
            Here's what has happened so far:
            {conversation_summary}

            Current user preferences:
            {prefs_text}

            Available recipe options:
            {recipe_summaries}

            The user has expressed interest in this type of dish but hasn't selected a specific recipe yet.
            Present these recipe options in a friendly, conversational way.
            DO NOT include full ingredients or directions - just describe each option briefly and invite them to choose one.
            Ask which recipe they'd like to see, or offer to show them the full recipe for one of these options.
            Be warm and helpful, but wait for them to select a specific recipe before providing full details.
            """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                temperature=0.8,
            )

            return {"generated_response": response.choices[0].message.content.strip()}

        except Exception as e:
            print("❌ LLM response generation failed:", e)
            # Fallback: simple list of options
            options_text = "\n".join([
                f"- {r.get('title', 'Untitled Recipe')}"
                for r in results[:5]
            ])
            return {"generated_response": f"Here are some great options for you:\n\n{options_text}\n\nWhich one would you like the full recipe for?"}

    # Fallback: if no recipes found, return error message
    return {"generated_response": "Sorry, I couldn't find any recipes matching your preferences right now."}
