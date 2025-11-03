# agents/response_agent.py

import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🧠 --- Helper: summarize conversation context ---
def summarize_conversation(messages: List[Dict]) -> str:
    """Create a short narrative summary of the user's evolving preferences."""
    if not messages:
        return "The conversation just started. The user is making an initial request for a meal."

    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    joined = "\n".join(user_turns[-6:])  # keep recent turns

    try:
        summary_prompt = f"""
        Summarize the following conversation turns into one short paragraph
        describing how the user's meal preferences have evolved.
        Avoid greetings or repetitive phrasing.

        Conversation:
        {joined}
        """

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise summarization assistant."},
                {"role": "user", "content": summary_prompt.strip()},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Conversation summarization failed:", e)
        return "User has shared several changing preferences for dinner ideas."

# 🕵️ --- Helper: detect if user explicitly asked for a recipe ---
def user_requested_full_recipe(messages: List[Dict], results: List[Dict] = None) -> bool:
    """Detects if the user asked for the full recipe."""
    if not messages:
        return False
    
    # Get the last user message (search backwards from the end)
    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "").lower()
            break
    
    if not last_user_msg:
        return False
    
    # Expanded triggers to catch more variations
    triggers = [
        "give me the recipe",
        "show me the recipe",
        "how do i make",
        "how to make",
        "full recipe",
        "recipe for",
        "would like the recipe",
        "want the recipe",
        "need the recipe",
        "can i have the recipe",
        "could you give me the recipe",
        "instructions",
        "preparation",
        "how to prepare",
        "how do you make",
        "recipe please",
        "the recipe",
    ]
    
    # Check if any trigger phrase appears in the message
    if any(trigger in last_user_msg for trigger in triggers):
        return True
    
    # Also check if user is asking for a specific recipe by name
    # This handles cases like "Yes, I would like the recipe for Italian Pizzeria-Style Meat Pie"
    if results:
        for recipe in results:
            recipe_title = recipe.get("title", "").lower().strip()
            if not recipe_title:
                continue
            
            # Normalize both strings for comparison (remove extra spaces, special chars)
            normalized_recipe = " ".join(recipe_title.split())
            normalized_msg = " ".join(last_user_msg.split())
            
            # Check if the recipe title appears in the user's message (with some flexibility)
            # Try exact match first, then try matching key words
            if normalized_recipe in normalized_msg:
                # Make sure it's in a request context (contains words like "recipe", "for", "yes", etc.)
                request_words = ["recipe", "for", "yes", "please", "give", "show", "want", "like"]
                if any(word in normalized_msg for word in request_words):
                    return True
            
            # Also try matching key words from the recipe title (for partial matches)
            recipe_words = [w for w in normalized_recipe.split() if len(w) > 3]  # Skip short words
            if len(recipe_words) >= 2:  # Need at least 2 significant words
                matched_words = sum(1 for word in recipe_words if word in normalized_msg)
                if matched_words >= 2:  # If at least 2 key words match
                    request_words = ["recipe", "for", "yes", "please", "give", "show", "want", "like"]
                    if any(word in normalized_msg for word in request_words):
                        return True
    
    return False

# 🍽️ --- Helper: format a single recipe cleanly ---
def format_recipe(recipe: dict) -> str:
    """
    Formats a single recipe into a readable, friendly markdown block.
    Includes emojis for readability, and ends with source attribution.
    """
    title = recipe.get("title", "Untitled Recipe")
    ingredients = recipe.get("ingredients", [])
    directions = recipe.get("directions", [])
    source = recipe.get("source", "Unknown Source")
    link = recipe.get("link", "").strip()

    # Format ingredients with emoji bullets
    formatted_ingredients = "\n".join([f"- 🧂 {item}" for item in ingredients]) if ingredients else "No ingredients provided."

    # Format directions with step numbers and icons
    formatted_directions = "\n".join(
        [f"{idx + 1}. 🔪 {step}" for idx, step in enumerate(directions)]
    ) if directions else "No instructions provided."

    # Format source attribution
    source_attribution = f"\n\n📖 *Source: [{source}]({link})*" if link else f"\n\n📖 *Source: {source}*"

    return (
        f"### 🍽️ {title}\n\n"
        f"**Ingredients:**\n{formatted_ingredients}\n\n"
        f"**Directions:**\n{formatted_directions}"
        f"{source_attribution}"
    )


# 🧩 --- Main Response Agent ---
@tool
def generate_response(preferences: Dict, results: List[Dict], messages: List[Dict] = []) -> Dict:
    """
    Generate a conversational, multilingual meal recommendation based on
    user preferences, search results, and summarized chat history.
    """

    language = preferences.get("language", "English")
    conversation_summary = summarize_conversation(messages)
    wants_recipe = user_requested_full_recipe(messages, results)

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

    # If user explicitly requested a recipe, show the full recipe with ingredients and directions
    top_recipe: Optional[Dict] = None
    
    if wants_recipe:
        # Initialize with top recipe if available
        if results:
            top_recipe = results[0]
        
        # Try to find the specific recipe the user mentioned
        if messages:
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
                        break
                    
                    # Try matching key words from the recipe title
                    recipe_words = [w for w in normalized_recipe.split() if len(w) > 3]
                    if len(recipe_words) >= 2:
                        matched_words = sum(1 for word in recipe_words if word in normalized_msg)
                        if matched_words >= 2:
                            top_recipe = recipe
                            break
        
        # If no specific recipe found, use top recipe
        if not top_recipe and results:
            top_recipe = results[0]
        
        # MANDATORY: Format and return full recipe with ingredients and directions
        if top_recipe:
            formatted_recipe = format_recipe(top_recipe)
            intro = (
                f"Here's the full recipe for **{top_recipe.get('title', '').strip()}**.\n"
            )
            return {"generated_response": intro + formatted_recipe}
    
    # Otherwise, show recipe options WITHOUT full details (user hasn't selected one yet)
    if results:
        recipe_summaries = "\n\n".join([
            f"{i+1}. **{r['title']}** — made with {', '.join(r['ingredients'][:3]) if r.get('ingredients') else 'various ingredients'}..."
            for i, r in enumerate(results[:5])  # Show top 5 options
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
