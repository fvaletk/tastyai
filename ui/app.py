# ui/app.py

import streamlit as st
import requests
import os
import html
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="TastyAI Chat", layout="centered")

# --- CSS Styling ---
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .title-bar {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.6rem;
            font-weight: bold;
            padding: 10px;
            background-color: #FAF3E0;
            border-bottom: 1px solid #DDD;
            margin-bottom: 20px;
            color: black;
        }
        .message-wrapper {
            display: flex;
            margin: 8px 0;
            width: 100%;
        }
        .message-wrapper.user {
            justify-content: flex-end;
        }
        .message-wrapper.assistant {
            justify-content: flex-start;
        }
        .chat-bubble {
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 80%;
            word-wrap: break-word;
            color: black;
        }
        .user .chat-bubble {
            background-color: #DCF8C6;
        }
        .assistant .chat-bubble {
            background-color: #F1F0F0;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 60vh;
            overflow-y: auto;
            padding: 10px 15px;
            border: 1px solid #EEE;
            border-radius: 10px;
            background-color: #FFFFFF;
        }
        .reset-button {
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown('<div class="title-bar">🍕 TastyAI - Your Personal Recipe Assistant</div>', unsafe_allow_html=True)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = ""
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None
if "loading" not in st.session_state:
    st.session_state.loading = False

# --- Load Chat History ---
if not st.session_state.history_loaded:
    try:
        if st.session_state.conversation_id:
            # Load specific conversation history
            history_response = requests.get(
                f"{BACKEND_URL}/chat/history/{st.session_state.conversation_id}",
                timeout=10
            )
        else:
            # Load all messages from all conversations
            history_response = requests.get(
                f"{BACKEND_URL}/chat/history",
                timeout=10
            )
        
        if history_response.status_code == 200:
            history_data = history_response.json()
            # Convert Message objects to dict format
            st.session_state.messages = [
                {
                    "role": msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", ""),
                    "content": msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                }
                for msg in history_data.get("messages", [])
            ]
            st.session_state.history_loaded = True
    except Exception as e:
        st.warning(f"Could not load chat history: {e}")

# --- New Chat Button ---
col1, col2 = st.columns([0.3, 0.7])
with col1:
    if st.button("🗑️ New Chat", width="content"):
        # st.info('This is a purely informational message', icon="ℹ️")
        st.session_state.messages = []
        st.session_state.conversation_id = ""
        st.session_state.history_loaded = False
        st.session_state.pending_message = None
        st.session_state.loading = False
        st.rerun()

# --- Chat History Display ---
chat_container = st.container()
with chat_container:
    if st.session_state.messages:
        for msg in st.session_state.messages:
            # Handle both dict and object formats
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "assistant")
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            
            role_class = "user" if role == "user" else "assistant"
            # Escape HTML in content to prevent XSS
            escaped_content = html.escape(content)
            bubble = f'<div class="message-wrapper {role_class}"><div class="chat-bubble">{escaped_content}</div></div>'
            st.markdown(bubble, unsafe_allow_html=True)
    else:
        st.info("👋 Start a conversation by asking about recipes!")

# --- Input + Loading ---
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("Type your message", placeholder="What would you like to eat today?")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.info('Using user input: ' + user_input.strip(), icon="ℹ️")
    st.info('Using conversation id: ' + st.session_state.conversation_id, icon="ℹ️")
    # Store the pending message and add to UI
    user_message = {"role": "user", "content": user_input.strip()}
    st.session_state.messages.append(user_message)
    st.session_state.pending_message = user_input.strip()
    st.session_state.loading = True
    st.session_state.history_loaded = True  # Mark as loaded to prevent reload
    st.rerun()

# Process pending message
if st.session_state.pending_message and st.session_state.loading:
    user_input = st.session_state.pending_message
    st.session_state.pending_message = None  # Clear pending message
    
    try:
        # print("--------------------------------")       
        # print("USER INPUT: ", user_input)
        # print("CONVERSATION ID: ", st.session_state.conversation_id)
        # print("--------------------------------")
        response = requests.post(
            f"{BACKEND_URL}/recommend",
            json={
                "message": user_input,
                "conversation_id": st.session_state.conversation_id or ""
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            # Update conversation_id
            st.session_state.conversation_id = data.get("conversation_id", "")
            # print("#################################")
            # print("DATA: ", data)
            # print("CONVERSATION ID: ", st.session_state.conversation_id)
            # print("#################################")
            
            # Convert messages from response to dict format
            response_messages = data.get("messages", [])
            st.session_state.messages = [
                {
                    "role": msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", ""),
                    "content": msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                }
                for msg in response_messages
            ]
        else:
            error_msg = response.json().get("detail", "Sorry, something went wrong on the server.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {error_msg}"
            })

    except requests.exceptions.RequestException as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Error reaching backend: {str(e)}"
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Unexpected error: {str(e)}"
        })
    
    st.session_state.loading = False
    st.rerun()

# --- Loading spinner ---
if st.session_state.loading:
    with st.spinner("Thinking..."):
        st.empty()