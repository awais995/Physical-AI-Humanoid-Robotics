from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid


class RetrievedPassage(BaseModel):
    """
    Model representing a passage retrieved from the vector database
    """
    id: str = str(uuid.uuid4())  # Unique identifier for this passage
    content: str  # The actual content of the passage
    source_url: str  # URL where the content was sourced from
    source_title: str  # Title of the source page
    source_section: Optional[str] = None  # Section/chapter name
    source_anchor: Optional[str] = None  # Specific anchor/heading within section
    similarity_score: float = 0.0  # Similarity score from vector search (0.0 to 1.0)
    passage_metadata: Dict = {}  # Additional metadata about the passage
    embedding_vector: Optional[List[float]] = None  # The embedding vector (optional)
    context_before: Optional[str] = None  # Context before the passage
    context_after: Optional[str] = None  # Context after the passage
    char_start_pos: int = 0  # Start position in original document
    char_end_pos: int = 0  # End position in original document

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"