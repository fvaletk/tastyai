from typing import Optional, Dict, List, Any
from pydantic import BaseModel

class TastyAIState(BaseModel):
    user_input: Optional[str] = None
    preferences: Optional[Dict[str, str]] = None
    results: Optional[List[Dict[str, Any]]] = None
    generated_response: Optional[str] = None
