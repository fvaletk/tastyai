from typing import Literal
from openai import OpenAI
from langchain_core.tools import tool
from pydantic import BaseModel

# Define schema for structured classification
class RecipeDecisionInput(BaseModel):
    message_history: str

class RecipeDecisionOutput(BaseModel):
    user_wants_recipe: Literal["yes", "no"]

# Register as LangChain-compatible tool
@tool("decide_recipe_request")
def decide_recipe_request(input: RecipeDecisionInput) -> RecipeDecisionOutput:
    """
    Determines whether the user is explicitly ready to receive a recipe now.
    Analyzes the full message history to detect clear intent (e.g., "yes I want to make it", "give me the recipe").
    """

    print("DECISION REQUEST INPUT: input", input)

    message_text = input.message_history.strip()
    if not message_text:
        return RecipeDecisionOutput(user_wants_recipe="no")

    messages = [
        {"role": "system", "content": "You are an intent detection assistant. Your job is to decide if the user is ready to receive a recipe based on the conversation history. Only respond with 'yes' or 'no'."},
        {"role": "user", "content": input.message_history}
    ]

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "decide_recipe_request",
                    "description": "Detects if the user wants to receive a full recipe now.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_wants_recipe": {
                                "type": "string",
                                "enum": ["yes", "no"]
                            }
                        },
                        "required": ["user_wants_recipe"]
                    }
                }
            }
        ],
        tool_choice={"type": "function", "function": {"name": "decide_recipe_request"}},
        temperature=0.0
    )

    # Parse result
    function_args = response.choices[0].message.tool_calls[0].function.arguments
    result = RecipeDecisionOutput.model_validate_json(function_args)
    print("DECISION RESULT: result", result)
    return result
