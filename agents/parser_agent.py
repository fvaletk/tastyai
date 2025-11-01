import os
import json
from typing import List
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from mcp.schema import Message
from models.schema import UserPreferences

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
    Extract user preferences using OpenAI Function Calling for structured output.
    """
    # Define the tool/function schema for preferences
    function_spec = {
        "name": "extract_preferences",
        "description": "Extract structured meal preferences from user input",
        "parameters": UserPreferences.model_json_schema()
    }

    # Convert Message objects to dicts
    messages_payload = [msg.dict() for msg in messages]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You extract structured meal preferences from user messages. Always return your reply as a function call."}
            ] + messages_payload,
            tools=[{"type": "function", "function": function_spec}],
            tool_choice="auto",  # Let the model choose to call the function
        )

        tool_call = response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments

        # Parse tool_call arguments to a dict
        import json
        parsed = json.loads(args)
        return parsed

    except Exception as e:
        print("⚠️ ParserAgent function call failed:", e)
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
