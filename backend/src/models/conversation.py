from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from .query import Query
from .response import Response


class Conversation(BaseModel):
    """
    Model representing a conversation between user and chatbot
    """
    id: Optional[str] = None  # Generated UUID
    user_id: Optional[str] = None  # Optional user identifier
    title: str  # Auto-generated title from first query
    created_at: datetime
    updated_at: datetime
    queries: List[Query] = []
    responses: List[Response] = []
    active: bool = True  # Whether the conversation is still active
    metadata: dict = {}  # Additional metadata about the conversation

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"