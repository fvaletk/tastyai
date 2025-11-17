# mcp/helpers.py

from agents.services import is_follow_up_question

def is_follow_up_question(messages: list) -> bool:
    """
    Detect if the current user message is a follow-up question rather than a new search request.
    
    Returns True if:
    - There are previous messages in the conversation
    - The latest user message contains follow-up/comparison keywords
    - The user is not making a completely new request
    
    Args:
        messages: List of conversation messages
        
    Returns:
        bool: True if this is a follow-up question, False if it's a new search
    """
    if not messages or len(messages) < 2:
        # First message in conversation - not a follow-up
        return False
    
    # Get the last user message
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "").lower()
                break
        else:
            if msg.role == "user":
                last_user_msg = msg.content.lower()
                break
    
    if not last_user_msg:
        return False
    
    # Follow-up question indicators
    follow_up_keywords = [
        # Comparison questions
        "which one", "which is", "what is the difference", "what's the difference",
        "compare", "comparison", "versus", "vs", "between",
        
        # Time/difficulty questions
        "how long", "how much time", "takes less", "takes more", "quicker", "faster",
        "slower", "easier", "harder", "simpler", "more difficult",
        
        # Nutritional questions
        "healthier", "more protein", "less carbs", "fewer carbs", "calories",
        
        # Selection/preference questions
        "better", "prefer", "recommend", "which would you",
        
        # Recipe request phrases
        "give me the recipe", "show me the recipe", "i'll take", "i want that one",
        "the first one", "the second one", "the third one", "recipe for",
        "how do i make", "how to make", "instructions for",
        
        # Follow-up indicators
        "what about", "tell me more", "more details", "more info",
        "anything else", "other options", "alternatives"
    ]
    
    # Check if message contains follow-up keywords
    if any(keyword in last_user_msg for keyword in follow_up_keywords):
        return True
    
    # Check if the message is very short (likely a follow-up)
    # e.g., "the first one", "yes", "that one"
    word_count = len(last_user_msg.split())
    if word_count <= 5:
        # Short messages with selection/affirmation words
        short_follow_up_words = ["yes", "no", "sure", "okay", "thanks", "one", "that", "this", "it"]
        if any(word in last_user_msg for word in short_follow_up_words):
            return True
    
    # If none of the above, assume it's a new search request
    return False