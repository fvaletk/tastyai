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
def user_requested_full_recipe(messages: List[Dict]) -> bool:
    """Detects if the user asked for the full recipe."""
    if not messages:
        return False
    last_user_msg = messages[-1]["content"].lower()
    triggers = [
        "give me the recipe",
        "show me the recipe",
        "how do i make",
        "full recipe",
        "recipe for",
        "instructions",
        "preparation",
        "how to prepare",
    ]
    return any(trigger in last_user_msg for trigger in triggers)

# 🧩 --- Main Response Agent ---
@tool
def generate_response(preferences: Dict, results: List[Dict], messages: List[Dict] = []) -> Dict:
    """
    Generate a conversational, multilingual meal recommendation based on
    user preferences, search results, and summarized chat history.
    """

    language = preferences.get("language", "English")
    conversation_summary = summarize_conversation(messages)
    wants_recipe = user_requested_full_recipe(messages)

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

    # 🧭 Find the top recipe if needed
    top_recipe: Optional[Dict] = results[0] if results else None

    # If user asked for full recipe, format and return it directly
    if wants_recipe and top_recipe:
        formatted_recipe = format_recipe(top_recipe)
        intro = (
            f"Great choice! Here's the full recipe for your {top_recipe.get('title', '').strip()}.\n"
            "It’s a delicious option that balances flavor and heartiness — perfect for your dinner tonight.\n\n"
        )
        return {"generated_response": intro + formatted_recipe}

    # Otherwise, conversational response
    recipe_summaries = "\n\n".join([
        f"{i+1}. **{r['title']}** — made with {', '.join(r['ingredients'][:3])}..."
        for i, r in enumerate(results)
    ]) if results else "No relevant recipes found yet."

    user_prompt = f"""
        Here's what has happened so far:
        {conversation_summary}

        Current user preferences:
        {prefs_text}

        Available recipes:
        {recipe_summaries}

        Based on all this, continue the conversation naturally.
        If the user seems close to deciding, gently guide them or offer the next step.
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

        generated = response.choices[0].message.content.strip()
        return {"generated_response": generated}

    except Exception as e:
        print("❌ LLM response generation failed:", e)
        return {"generated_response": "Sorry, I couldn’t generate a response right now."}
