# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from mcp.graph import build_graph
from mcp.schema import TastyAIState
from models.schema import Message, MessageRequest, PreferencesResponse, RecipeMatch
from uuid import uuid4
from db.services import load_conversation_history, load_all_conversation_history, delete_all_conversation_history
import json

app = FastAPI(title="TastyAI API", version="0.1")

# Build LangGraph MCP server
graph = build_graph()

# In-memory storage for conversation results (use Redis in production)
conversation_results = {}

@app.get("/chat/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Get chat history for a conversation."""
    try:
        messages = load_conversation_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages": [Message(**msg) for msg in messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading chat history: {str(e)}")

@app.get("/chat/history")
async def get_all_chat_history():
    """Get all chat history from all conversations."""
    try:
        messages = load_all_conversation_history()
        return {
            "conversation_id": None,
            "messages": [Message(**msg) for msg in messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading all chat history: {str(e)}")

@app.delete("/chat/history/all")
async def delete_chat_history():
    """Delete all chat history."""
    try:
        delete_all_conversation_history()
        return {"message": "All chat history deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat history: {str(e)}")

@app.post("/recommend", response_model=PreferencesResponse)
async def recommend_meal(request: MessageRequest):
    print("######################################################")
    print("NEW INCOMING REQUEST")
    print("######################################################")

    user_input = request.message
    conversation_id = request.conversation_id or str(uuid4())

    try:
        db_messages = load_conversation_history(conversation_id)
        db_messages.append({"role": "user", "content": user_input})

        messages = [Message(**msg) for msg in db_messages]

        previous_results = conversation_results.get(conversation_id, None)

        print("######################################################")
        print(f"💾 LOADING PREVIOUS RESULTS FOR CONVERSATION {conversation_id}: {len(previous_results) if previous_results else 0} recipes")
        print("######################################################")

        initial_state = TastyAIState(
            user_input=user_input,
            conversation_id=conversation_id,
            messages=messages,
            results=previous_results
        )

        result = graph.invoke(initial_state)

        result["messages"].append(
            Message(
                role="assistant",
                content=result.get("generated_response")
            )
        )

        if not result.get('preferences'):
            raise HTTPException(status_code=500, detail="Parser failed to extract preferences.")
        if not result.get('results'):
            raise HTTPException(status_code=500, detail="Search agent failed to find recipes.")

        conversation_results[conversation_id] = result['results']
        print(f"💾 Stored {len(result['results'])} recipes for conversation {conversation_id}")

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
            messages=result['messages'],
            conversation_id=conversation_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
