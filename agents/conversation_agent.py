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
    
    # Check if recipes have been shown (indicated by assistant messages)
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    recipes_have_been_shown = len(assistant_messages) > 0
    
    # Get conversation context (last 4 messages)
    recent_messages = messages[-4:] if len(messages) > 4 else messages
    context = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent_messages])
    
    # Count user messages to determine if this is likely the first request
    user_messages = [m for m in messages if m.get("role") == "user"]
    is_first_request = len(user_messages) == 1
    
    prompt = f"""
    Analyze the user's latest message and classify their intent.

    Conversation context:
    {context}

    User's latest message: "{last_user_msg}"

    IMPORTANT CONTEXT:
    - Recipes have been shown in this conversation: {recipes_have_been_shown}
    - This is the first user message: {is_first_request}
    - Number of assistant responses: {len(assistant_messages)}

    Classify the intent into ONE of these categories:

    1. "new_search" - User wants to search for new recipes (new cuisine, new dish, different meal)
       - Use this if NO recipes have been shown yet (first message or no assistant responses)
       - Use this if user is asking for something completely new
       - Examples: "I want Italian food", "Show me breakfast recipes", "I'm looking for desserts", "I want to prepare a pie" (if no recipes shown yet)

    2. "comparison" - User is comparing or asking about differences between previously shown recipes
       - ONLY use if recipes have already been shown
       - Examples: "What's the difference?", "Which one is quicker?", "Which is healthier?"

    3. "recipe_request" - User wants the full recipe for a specific dish that was already mentioned/shown
       - ONLY use if recipes have already been shown and user is selecting one
       - Examples: "Give me the recipe for X" (where X was shown), "I want to prepare the Lasagna" (where Lasagna was shown), "Show me the first one"

    4. "general" - General questions or conversation
       - Examples: "Thanks", "Tell me more", "What do you recommend?"

    CRITICAL RULE: If recipes have NOT been shown yet (no assistant messages), the intent MUST be "new_search" 
    even if the user says "I want to prepare X" or "give me the recipe for X". They need to see recipe options first.

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
        
        # Post-processing: Override recipe_request if no recipes have been shown
        if intent == "recipe_request" and not recipes_have_been_shown:
            print("⚠️ Overriding recipe_request to new_search: No recipes have been shown yet")
            intent = "new_search"
            reasoning = f"Overridden: {reasoning} (no recipes shown yet)"
        
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
