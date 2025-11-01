# agents/response_agent.py

import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def generate_response(preferences: dict, results: list) -> dict:
    """
    Generates a friendly response using GPT based on user preferences and recipe search results.
    """

    language = preferences.get("language", "en")

    system_prompt = f"""
    You are a multilingual, helpful recipe assistant.

    Automatically detect the user's preferred language based on the conversation context and reply fully in this language: **{language}**.

    Use a friendly, natural tone. Make your suggestions based on the user's stated preferences and the list of matching recipes.

    Be conversational, warm, and specific — and always use the same language the user used: **{language}**.
    """

    # Format the prompt
    intro = "Generate a meal recommendation based on the following:\n"
    prefs = (
        f"Preferences:\n"
        f"- Cuisine: {preferences.get('cuisine')}\n"
        f"- Diet: {preferences.get('diet')}\n"
        f"- Dish: {preferences.get('dish')}\n"
        f"- Ingredients: {', '.join(preferences.get('ingredients', [])) or 'None'}\n"
        f"- Allergies: {', '.join(preferences.get('allergies', [])) or 'None'}\n"
        f"- Meal type: {preferences.get('meal_type')}\n"
        f"- Cooking time: {preferences.get('cooking_time')}\n"
    )

    recipe_list = "\n".join(
        [f"- {r['title']} (link: {r['link']})" for r in results]
    )

    prompt = intro + prefs + "\nRecipes:\n" + recipe_list

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": prompt.strip()}
            ]
        )
        generated = response.choices[0].message.content
        return {"generated_response": generated}
    except Exception as e:
        print("❌ LLM response generation failed:", e)
        return {"generated_response": f"Sorry, I couldn't generate a recommendation."}

