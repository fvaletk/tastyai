# agents/helpers.py

import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Dict, List, Optional
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_to_english(text: str, source_language: str) -> str:
    """Translate a message to English for intent detection."""
    try:
        prompt = f"Translate the following {source_language} message into English, keeping the meaning intact:\n\n{text}"

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Translation failed:", e)
        return text  # Fallback: use original

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
def user_requested_full_recipe(messages: List[Dict], results: List[Dict] = None, language: str = "English", is_comparison: bool = False) -> bool:
    """
    Detects if the user asked for the full recipe.
    
    Args:
        messages: Conversation history
        results: Available recipe results
        language: User's language
        is_comparison: If True, this is already identified as a comparison question (skip recipe check)
    """
    # print("******************************************************")
    # print("MESSAGES: ", messages)
    # print("******************************************************")
    # print("RESULTS: ", results)
    # print("******************************************************")
    # print("LANGUAGE: ", language)
    # print("******************************************************")
    
    if not messages:
        return False
    
    # 🔒 CRITICAL: If this is already a comparison question, DON'T treat it as a recipe request
    if is_comparison:
        return False

    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    translated_msg = translate_to_english(last_user_msg.lower(), language)
    
    # 🚫 ANTI-PATTERNS: If message contains comparison/question words, likely NOT a recipe request
    comparison_anti_patterns = [
        "which one", "which is", "what is", "what's the", "how does", "compare",
        "difference", "versus", "vs", "between", "or", "better", "prefer"
    ]
    
    if any(pattern in translated_msg for pattern in comparison_anti_patterns):
        # Only continue if there's an explicit recipe request despite the comparison words
        explicit_recipe_request = any(phrase in translated_msg for phrase in [
            "give me the recipe for", "show me the recipe for", "i want the recipe",
            "recipe please", "full recipe"
        ])
        if not explicit_recipe_request:
            return False

    # ✅ Explicit recipe request triggers
    triggers = [
        "give me the recipe", "show me the recipe", "how do i make", "how to make",
        "full recipe", "recipe for", "would like the recipe", "want the recipe",
        "need the recipe", "can i have the recipe", "instructions for",
        "how to prepare", "recipe please", "i'll take", "i choose",
        "let's go with", "i'd like to try"
    ]

    if any(trigger in translated_msg for trigger in triggers):
        return True
    
    # ✅ User selects by number or position
    selection_patterns = [
        "the first one", "the second one", "the third one",
        "first recipe", "second recipe", "third recipe",
        "number 1", "number 2", "number 3", "#1", "#2", "#3",
        "option 1", "option 2", "option 3"
    ]
    
    if any(pattern in translated_msg for pattern in selection_patterns):
        return True

    # ✅ Check if user mentioned a specific recipe title with selection intent
    if results:
        for recipe in results:
            title = recipe.get("title", "").lower().strip()
            if not title:
                continue

            normalized_title = " ".join(title.split())
            
            # Only match if the FULL title is in the message AND there's selection intent
            if normalized_title in translated_msg:
                selection_intent = any(word in translated_msg for word in [
                    "yes", "please", "this one", "that one", "sounds good",
                    "i'll have", "i want", "give me", "show me"
                ])
                if selection_intent:
                    return True

    return False

def is_follow_up_comparison(messages: List[Dict], results: List[Dict], language: str = "English") -> bool:
    """
    Detect if the user is asking a follow-up comparison question.
    Enhanced to catch more comparison scenarios including time-related questions.
    """
    if not messages or not results:
        return False

    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    translated_msg = translate_to_english(last_user_msg.lower(), language)

    # Expanded list of comparison keywords
    comparison_keywords = [
        # Direct comparison words
        "compare", "comparison", "which one", "which is", "difference", "different",
        "vs", "versus", "between",
        
        # Nutritional comparisons
        "healthier", "more protein", "fewer carbs", "less carbs", "more carbs",
        "calories", "fat content", "nutritious",
        
        # Time-related comparisons
        "quicker", "faster", "slower", "longer", "shorter",
        "prep time", "cooking time", "preparation time",
        "less time", "more time", "takes less", "takes more",
        "how long", "time to prepare", "time to cook",
        
        # Difficulty comparisons
        "easier", "simpler", "harder", "more difficult", "complicated",
        
        # Cost and alternatives
        "cheaper", "expensive", "alternative", "option", "substitute",
        
        # Preference and recommendation
        "better", "prefer", "recommend", "suggest", "best",
        
        # Referencing specific options
        "first one", "second one", "third one", "first option", "second option"
    ]

    # Check if any comparison keyword is present
    if any(kw in translated_msg for kw in comparison_keywords):
        return True
    
    # Additional context check: if user references multiple recipes by name
    if results and len(results) >= 2:
        recipe_titles_mentioned = 0
        for recipe in results[:3]:  # Check top 3 recipes
            title = recipe.get("title", "").lower().strip()
            if title and title in translated_msg:
                recipe_titles_mentioned += 1
        
        # If user mentions 2+ recipe titles, likely comparing
        if recipe_titles_mentioned >= 2:
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

def handle_comparative_response(results: List[Dict], user_msg: str, language: str, messages: List[Dict] = None) -> Dict:
    """
    Responds to user follow-up questions comparing or refining recipe suggestions.
    Enhanced to provide actual data from recipe directions for accurate time comparisons.
    """
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

def user_selected_recipe(messages: List[Dict], results: List[Dict], language: str = "English") -> Optional[Dict]:
    """
    Try to determine if the user selected one of the recipes by name or number.
    Return the matching recipe dict if found, else None.
    """
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "").strip()
    translated_msg = translate_to_english(last_user_msg.lower(), language)

    # Check for numbered selection (e.g., "I like the first one", "option 2", etc.)
    number_map = {
        "first": 0,
        "1": 0,
        "one": 0,
        "second": 1,
        "2": 1,
        "two": 1,
        "third": 2,
        "3": 2,
        "three": 2
    }

    for word, index in number_map.items():
        if word in translated_msg and len(results) > index:
            return results[index]

    # GPT fallback — use model to resolve which title user meant
    titles = [r["title"] for r in results]
    matched_title = resolve_recipe_reference(last_user_msg, titles, language)
    if matched_title:
        for recipe in results:
            if recipe["title"].strip().lower() == matched_title.strip().lower():
                return recipe

    return None

def translate_text(text: str, target_language: str) -> str:
    try:
        prompt = f"Please translate the following text into {target_language}, keeping formatting like bullets and line breaks intact:\n\n{text}"
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a careful translator that preserves Markdown formatting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Translation failed:", e)
        return text

def resolve_recipe_reference(user_message: str, recipe_titles: List[str], language: str) -> Optional[str]:
    """Use GPT to determine which recipe the user is referring to by title."""
    try:
        formatted_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(recipe_titles)])

        prompt = f"""
        The user said: "{user_message}"

        Here are the available recipe options:
        {formatted_titles}

        Based on their message, which recipe are they referring to? Respond with the exact title. 
        If no clear match, just say "None".
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant interpreting recipe choices. The user speaks {language}."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2
        )

        reply = response.choices[0].message.content.strip()
        if reply.lower() == "none":
            return None
        return reply
    except Exception as e:
        print("⚠️ Failed to resolve recipe reference:", e)
        return None
