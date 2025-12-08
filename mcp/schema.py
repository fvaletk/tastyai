# mcp/schema.py

from typing import Optional, List
from pydantic import BaseModel, Field
from models.schema import UserPreferences, RecipeMatch, Message

class TastyAIState(BaseModel):
    user_input: Optional[str] = None
    conversation_id: Optional[str] = None
    preferences: Optional[UserPreferences] = None
    results: Optional[List[RecipeMatch]] = None
    previous_results: Optional[List[RecipeMatch]] = None
    generated_response: Optional[str] = None
    intent: Optional[str] = None
    intent_reasoning: Optional[str] = None  # Reasoning from intent classification
    recipe_request_type: Optional[str] = None  # "specific_recipe" | "dish" | "new_dish"
    matched_recipe_title: Optional[str] = None
    dish_name: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
