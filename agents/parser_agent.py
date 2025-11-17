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
            tool_choice={"type": "function", "function": {"name": "extract_preferences"}},  # Force function call
        )

        # Check if response has choices
        if not response.choices:
            raise ValueError("No choices in API response")
        
        message = response.choices[0].message
        
        # Check if tool_calls exists and is not None/empty
        if not message.tool_calls:
            print("⚠️ ParserAgent: No tool_calls in response. Message content:", message.content)
            raise ValueError("No tool_calls in response")
        
        if len(message.tool_calls) == 0:
            print("⚠️ ParserAgent: Empty tool_calls list. Message content:", message.content)
            raise ValueError("Empty tool_calls list")
        
        tool_call = message.tool_calls[0]
        
        # Check if tool_call has function and arguments
        if not tool_call.function:
            raise ValueError("Tool call missing function attribute")
        
        args = tool_call.function.arguments
        
        if not args:
            raise ValueError("Tool call function missing arguments")

        # Parse tool_call arguments to a dict
        parsed = json.loads(args)
        print("######################################################")
        print("PARSING", parsed)
        print("######################################################")
        return parsed

    except Exception as e:
        print("⚠️ ParserAgent function call failed:", e)
        print(f"⚠️ Error type: {type(e).__name__}")
        import traceback
        print(f"⚠️ Traceback: {traceback.format_exc()}")
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
