# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from mcp.graph import build_graph
from mcp.schema import TastyAIState

app = FastAPI(title="TastyAI API", version="0.1")

# Build LangGraph MCP server
graph = build_graph()

# Request schema
class MessageRequest(BaseModel):
    message: str

class RecipeMatch(BaseModel):
    title: str
    link: str
    ingredients: List[str]
    directions: List[str]
    source: str
    score: float

# Response schema
class PreferencesResponse(BaseModel):
    language: str
    cuisine: str
    diet: str
    results: List[RecipeMatch]
    generated_response: Optional[str]

@app.post("/recommend", response_model=PreferencesResponse)
async def recommend_meal(request: MessageRequest):
    user_input = request.message

    try:
        # Initial graph state
        initial_state = TastyAIState(user_input=user_input)

        # Run through LangGraph
        result = graph.invoke(initial_state)

        if not result.get('preferences'):
            raise HTTPException(status_code=500, detail="Parser failed to extract preferences.")
        if not result.get('results'):
            raise HTTPException(status_code=500, detail="Search agent failed to find recipes.")

        # Combine preferences and search results
        response_data = {
            "language": result["preferences"].get('language', 'unknown'),
            "cuisine": result["preferences"].get('cuisine', 'unknown'),
            "diet": result["preferences"].get('diet', 'unknown'),
            "results": result.get("results", []),
            "generated_response": result.get("generated_response", None)
        }

        return PreferencesResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
