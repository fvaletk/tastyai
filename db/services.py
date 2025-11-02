# db/services.py

from sqlalchemy.orm import Session
from db.database import SessionLocal
from db.models import ChatMessage

def save_message_to_db(conversation_id: str, role: str, content: str):
    session = SessionLocal()
    try:
        msg = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        session.add(msg)
        session.commit()
    except Exception as e:
        print(f"⚠️ Failed to save message: {e}")
        session.rollback()
    finally:
        session.close()

def load_conversation_history(conversation_id: str):
    session: Session = SessionLocal()
    try:
        messages = session.query(ChatMessage).filter(
            ChatMessage.conversation_id == conversation_id
        ).order_by(ChatMessage.timestamp.asc()).all()

        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    finally:
        session.close()
