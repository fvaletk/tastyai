import os
import json
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful assistant that extracts user meal preferences from natural language input.
Always return a JSON object with the following fields:
- language: the detected language code (e.g., 'en', 'es')
- cuisine: type of cuisine requested, e.g., 'italian', 'mexican', etc.
- diet: dietary constraint (e.g., 'low-carb', 'vegetarian', 'gluten-free', or 'none')

If any field is not mentioned, mark it as 'unknown'.

Format your response as valid JSON only, with no explanation or extra text.

Examples:

Input: "Hola, quiero una cena italiana baja en carbohidratos"
Output:
{
  "language": "es",
  "cuisine": "italian",
  "diet": "low-carb"
}

Input: "I'm looking for something vegetarian and Mexican."
Output:
{
  "language": "en",
  "cuisine": "mexican",
  "diet": "vegetarian"
}
"""

@tool
def parse_user_input(input_text: str) -> dict:
    """
    Extract user preferences from natural language text using OpenAI.
    Returns a dictionary with fields: language, cuisine, and diet.
    """
    user_msg = {"role": "user", "content": input_text}
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or gpt-3.5-turbo
            messages=[system_msg, user_msg],
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
            "diet": "unknown"
        }
