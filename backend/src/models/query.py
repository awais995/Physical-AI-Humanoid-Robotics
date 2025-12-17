from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from enum import Enum


class QueryMode(str, Enum):
    GLOBAL = "global"  # Query entire book
    SELECTED_TEXT_ONLY = "selected_text_only"  # Query only provided text selection


class Query(BaseModel):
    """
    Model representing a user query to the RAG system
    """
    id: Optional[str] = None  # Generated UUID
    conversation_id: str  # Associated conversation
    query_text: str  # The actual query from user
    query_mode: QueryMode = QueryMode.GLOBAL  # How to process the query
    selected_text: Optional[str] = None  # Text selected by user (if in selected_text_only mode)
    timestamp: datetime
    metadata: dict = {}  # Additional metadata about the query context
    filters: dict = {}  # Optional filters for retrieval (chapter, section, etc.)

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"