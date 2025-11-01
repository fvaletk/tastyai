import os
import json
from typing import List
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from mcp.schema import Message

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
    You are a helpful assistant that extracts structured meal preferences from natural language user input.

    Always return a valid JSON object with the following fields:
    - language: detected language code (e.g., "en", "es")
    - cuisine: type of cuisine (e.g., "italian", "mexican"). Use "unknown" if not mentioned.
    - diet: dietary restriction (e.g., "vegetarian", "gluten-free", "low-carb"). Use "unknown" if not mentioned.
    - dish: specific dish the user wants (e.g., "pizza", "lasagna"). Use "unknown" if not mentioned.
    - ingredients: list of desired ingredients (e.g., ["chicken", "avocado"]). Use an empty list if none are mentioned.
    - allergies: list of ingredients to avoid (e.g., ["gluten", "lactose"]). Use an empty list if none are mentioned.
    - meal_type: type of meal (e.g., "breakfast", "lunch", "dinner", "snack"). Use "unknown" if not mentioned.
    - cooking_time: estimated time required (e.g., "short", "medium", "long", or "unknown").

    Respond ONLY with a JSON object. Do not include any text before or after the JSON.
    Use "unknown" for any missing string fields and empty lists `[]` for missing array fields.

    Examples:

    Input: "I'm looking for a quick vegetarian Mexican dinner."
    Output:
    {
      "language": "en",
      "cuisine": "mexican",
      "diet": "vegetarian",
      "dish": "unknown",
      "ingredients": [],
      "allergies": [],
      "meal_type": "dinner",
      "cooking_time": "short"
    }

    Input: "Hola, tengo intolerancia a la lactosa y quiero una pizza italiana"
    Output:
    {
      "language": "es",
      "cuisine": "italian",
      "diet": "unknown",
      "dish": "pizza",
      "ingredients": [],
      "allergies": ["lactose"],
      "meal_type": "unknown",
      "cooking_time": "unknown"
    }
    """

@tool
def parse_user_input(messages: List[Message]) -> dict:
    """
    Extract user preferences from natural language text using OpenAI.
    Receives a list of Message objects and returns a dictionary with fields: language, cuisine, and diet.
    """
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    
    # Convert Message objects to dictionaries for OpenAI API
    messages_as_dicts = [msg.dict() for msg in messages]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or gpt-3.5-turbo
            messages=[system_msg] + messages_as_dicts,
            temperature=0,
        )

        raw_content = response.choices[0].message.content.strip()

        # Optional: Debug print
        print("🔍 LLM raw output:", raw_content)

        # Try to parse as JSON
        return json.loads(raw_content)

    except Exception as e:
        print("⚠️ ParserAgent error:", e)
        return {
            "language": "unknown",
            "cuisine": "unknown",
            "diet": "unknown",
            "dish": "unknown",
            "ingredients": [],
            "allergies": [],
            "meal_type": "unknown",
            "cooking_time": "unknown"
        }
