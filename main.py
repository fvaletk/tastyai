# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from mcp.graph import build_graph
from mcp.schema import TastyAIState
from models.schema import Message, MessageRequest, PreferencesResponse


app = FastAPI(title="TastyAI API", version="0.1")

# Build LangGraph MCP server
graph = build_graph()

@app.post("/recommend", response_model=PreferencesResponse)
async def recommend_meal(request: MessageRequest):
    user_input = request.message

    try:
        print("➡️ STEP 1: Building messages")
        messages = [
            Message(role="user", content=user_input)
        ]
        print("➡️ STEP 2: Creating initial state")
        initial_state = TastyAIState(
            user_input=user_input,
            messages=messages
        )
        print("➡️ STEP 3: Invoking graph")
        result = graph.invoke(initial_state)
        print("➡️ STEP 4: Appending assistant message")
        result["messages"].append(
            Message(
                role="assistant",
                content=result.get("generated_response")
            )
        )

        print("➡️ Final result keys:", result.keys())
        print("➡️ Preferences payload:", result['preferences'])

        if not result.get('preferences'):
            raise HTTPException(status_code=500, detail="Parser failed to extract preferences.")
        if not result.get('results'):
            raise HTTPException(status_code=500, detail="Search agent failed to find recipes.")

        # Build PreferencesResponse from state components
        # Handle both dict and UserPreferences object
        if hasattr(result['preferences'], 'dict'):
            preferences_dict = result['preferences'].dict()
        elif hasattr(result['preferences'], 'model_dump'):
            preferences_dict = result['preferences'].model_dump()
        else:
            preferences_dict = result['preferences']
        
        return PreferencesResponse(
            **preferences_dict,
            results=result['results'],
            generated_response=result.get('generated_response'),
            messages=result['messages']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
