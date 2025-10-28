import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def generate_response(preferences: dict, results: list) -> dict:
    """
    Generates a friendly response using GPT based on user preferences and recipe search results.
    """

    intro = "Generate a meal recommendation based on the following:\n"
    prefs = f"Preferences: Cuisine={preferences.get('cuisine')}, Diet={preferences.get('diet')}\n"
    recipe_list = "\n".join(
        [f"- {r['title']} (link: {r['link']})" for r in results]
    )
    prompt = intro + prefs + "Recipes:\n" + recipe_list

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a friendly recipe assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        generated = response.choices[0].message.content
        return {"generated_response": generated}
    except Exception as e:
        print("❌ LLM response generation failed:", e)
        return {"generated_response": "Sorry, I couldn't generate a recommendation."}
