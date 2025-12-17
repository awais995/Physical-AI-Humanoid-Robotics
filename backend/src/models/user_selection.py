from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class UserSelection(BaseModel):
    """
    Model representing a user's text selection and associated context
    """
    id: Optional[str] = None  # Generated UUID
    conversation_id: str  # Associated conversation
    selected_text: str  # The actual text selected by user
    selection_context: str  # Context around the selection (surrounding text)
    source_url: str  # URL where text was selected
    source_title: str  # Title of the source page
    source_section: Optional[str] = None  # Section/chapter where selection was made
    selection_start_pos: int  # Start position of selection in the text
    selection_end_pos: int  # End position of selection in the text
    timestamp: datetime
    metadata: dict = {}  # Additional metadata about the selection

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"