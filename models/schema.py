# models/schema.py

from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class MessageRequest(BaseModel):
    message: str

class RecipeMatch(BaseModel):
    title: str
    link: str
    ingredients: List[str]
    directions: List[str]
    source: str
    score: float

class UserPreferences(BaseModel):
    """Parsed user preferences from natural language input."""
    language: str
    cuisine: str
    diet: str
    dish: str
    ingredients: List[str]
    allergies: List[str]
    meal_type: str
    cooking_time: str

class PreferencesResponse(BaseModel):
    """Final API response including preferences, results, and messages."""
    language: str
    cuisine: str
    diet: str
    dish: str
    ingredients: List[str]
    allergies: List[str]
    meal_type: str
    cooking_time: str
    results: List[RecipeMatch]
    generated_response: Optional[str] = None
    messages: List[Message]
