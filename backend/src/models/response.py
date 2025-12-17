from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from .retrieved_passage import RetrievedPassage


class Response(BaseModel):
    """
    Model representing a response from the RAG system
    """
    id: Optional[str] = None  # Generated UUID
    conversation_id: str  # Associated conversation
    query_id: str  # Associated query
    response_text: str  # The generated response
    citations: List[RetrievedPassage] = []  # Passages used to generate response
    confidence: float = 0.0  # Confidence score (0.0 to 1.0)
    generated_at: datetime
    tokens_used: int = 0  # Number of tokens in the response
    processing_time_ms: float = 0.0  # Time taken to generate response
    hallucination_check_passed: bool = True  # Whether hallucination check passed
    metadata: dict = {}  # Additional metadata about the response

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"