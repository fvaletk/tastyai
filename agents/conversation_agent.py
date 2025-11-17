# agents/conversation_agent.py

import os
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Dict, List
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@tool
def classify_intent(messages: List[Dict]) -> Dict:
    """
    Classify user's intent: new_search, comparison, recipe_request, or general
    """
    
    if not messages or len(messages) < 1:
        return {"intent": "new_search"}
    
    # Get last user message
    last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    
    if not last_user_msg:
        return {"intent": "new_search"}
    
    # Get conversation context (last 4 messages)
    recent_messages = messages[-4:] if len(messages) > 4 else messages
    context = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent_messages])
    
    prompt = f"""
    Analyze the user's latest message and classify their intent.

    Conversation context:
    {context}

    User's latest message: "{last_user_msg}"

    Classify the intent into ONE of these categories:

    1. "new_search" - User wants to search for new recipes (new cuisine, new dish, different meal)
       Examples: "I want Italian food", "Show me breakfast recipes", "I'm looking for desserts"

    2. "comparison" - User is comparing or asking about differences between previously shown recipes
       Examples: "What's the difference?", "Which one is quicker?", "Which is healthier?"

    3. "recipe_request" - User wants the full recipe for a specific dish that was already mentioned
       Examples: "Give me the recipe for X", "I want to prepare the Lasagna", "Show me the first one"

    4. "general" - General questions or conversation
       Examples: "Thanks", "Tell me more", "What do you recommend?"

    Respond with ONLY a JSON object:
    {{
      "intent": "new_search" | "comparison" | "recipe_request" | "general",
      "confidence": 0.0 to 1.0,
      "reasoning": "brief explanation"
    }}

    CRITICAL: Your entire response must be valid JSON only. No other text.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intent classification expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=0.2
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up markdown code blocks if present
        result = result.replace("```json\n", "").replace("```\n", "").replace("```", "").strip()

        parsed = json.loads(result)
        
        intent = parsed.get("intent", "new_search")
        confidence = parsed.get("confidence", 0.5)
        reasoning = parsed.get("reasoning", "")
        
        print("######################################################")
        print(f"🎯 INTENT CLASSIFICATION: {intent} (confidence: {confidence})")
        print(f"💭 REASONING: {reasoning}")
        print("######################################################")
        
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning
        }
        
    except Exception as e:
        print(f"⚠️ Intent classification failed: {e}")
        # Fallback: if conversation has recipes, assume follow-up, else new search
        return {"intent": "new_search", "confidence": 0.3, "reasoning": "fallback"}