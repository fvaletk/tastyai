# agents/format_agent.py

from langchain_core.tools import tool
from typing import Dict

@tool
def format_recipe(recipe: Dict) -> str:
    """
    Formats a single recipe into a friendly markdown block with title, ingredients,
    directions, and source. Adds emojis and formatting for readability.
    """

    title = recipe.get("title", "Untitled Recipe")
    ingredients = recipe.get("ingredients", [])
    directions = recipe.get("directions", [])
    source = recipe.get("source", "Unknown Source")
    link = recipe.get("link", "").strip()

    # Format ingredients with emoji bullets
    formatted_ingredients = "\n".join(
        [f"- 🧂 {item}" for item in ingredients]
    ) if ingredients else "No ingredients provided."

    # Format directions with step numbers and icons
    formatted_directions = "\n".join(
        [f"{idx + 1}. 🔪 {step}" for idx, step in enumerate(directions)]
    ) if directions else "No instructions provided."

    # Source line
    source_line = f"\n\n📖 *Source: [{source}]({link})*" if link else f"\n\n📖 *Source: {source}*"

    print("FORMAT RECIPE RESULT: formatted_ingredients", formatted_ingredients)

    return (
        f"### 🍽️ {title}\n\n"
        f"**Ingredients:**\n{formatted_ingredients}\n\n"
        f"**Directions:**\n{formatted_directions}"
        f"{source_line}"
    )
