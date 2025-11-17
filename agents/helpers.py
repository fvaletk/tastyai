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
