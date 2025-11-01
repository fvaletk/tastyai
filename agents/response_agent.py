# agents/response_agent.py

import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def generate_response(preferences: dict, results: list) -> dict:
    """
    Generate a conversational, multilingual meal recommendation based on user preferences
    and recipe search results. Returns a friendly response that includes recipe details.
    """

    language = preferences.get("language", "en")

    # 🌍 Explicitly enforce response language
    system_prompt = f"""
    You are a multilingual, helpful culinary assistant named TastyAI.

    Always reply in this language: {language}.
    Be friendly, conversational, and natural.

    Your goal is to help the user discover the perfect meal by suggesting recipes, refining
    choices based on their comments, and, when appropriate, sharing the full recipe details
    (ingredients and directions).

    If the user hasn't chosen yet, offer 2–3 recipes in a helpful tone.
    If the user shows preference for one dish, provide its full recipe in detail.
    Never ask the user to click a link — summarize everything directly.
    """

    # 🧾 Build contextual input
    prefs_text = (
        f"Preferences:\n"
        f"- Cuisine: {preferences.get('cuisine')}\n"
        f"- Diet: {preferences.get('diet')}\n"
        f"- Dish: {preferences.get('dish')}\n"
        f"- Allergies: {', '.join(preferences.get('allergies', [])) or 'none'}\n"
        f"- Meal type: {preferences.get('meal_type')}\n"
        f"- Cooking time: {preferences.get('cooking_time')}\n"
    )

    # 🍽️ Build readable recipe list with details
    recipe_list = "\n\n".join([
        f"Recipe: {r['title']}\n"
        f"Ingredients:\n- " + "\n- ".join(r.get("ingredients", [])) + "\n"
        f"Directions:\n- " + "\n- ".join(r.get("directions", [])) + "\n"
        f"Source: {r.get('source')}"
        for r in results
    ])

    # 🎯 Combined prompt
    user_prompt = (
        "Based on the user's preferences and the following recipes, "
        "generate a conversational response and, if the user seems ready, "
        "include the complete recipe(s) directly in your message.\n\n"
        f"{prefs_text}\nAvailable recipes:\n{recipe_list}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.7,
        )

        generated = response.choices[0].message.content
        return {"generated_response": generated}

    except Exception as e:
        print("❌ LLM response generation failed:", e)
        return {
            "generated_response": f"Sorry, I couldn't generate a response at the moment."
        }
