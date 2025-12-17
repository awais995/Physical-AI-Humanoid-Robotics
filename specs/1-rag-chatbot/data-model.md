# Data Model: RAG Chatbot System

## Conversation Entity

**Description**: Represents a user's chat session with the system, containing metadata and message history

**Fields**:
- `id`: Unique identifier for the conversation
- `created_at`: Timestamp when conversation was initiated
- `updated_at`: Timestamp of last activity
- `session_token`: Optional session identifier (for non-authenticated users)
- `mode`: Query mode (global, chapter-section, selected-text-only)
- `messages`: Array of message objects (user queries and system responses)

**Validation rules**:
- ID must be unique
- Created_at must be before updated_at
- Mode must be one of the allowed values

## Query Entity

**Description**: A user's question or input to the chatbot system

**Fields**:
- `id`: Unique identifier for the query
- `conversation_id`: Reference to parent conversation
- `content`: The actual query text
- `timestamp`: When the query was submitted
- `query_mode`: The retrieval mode used for this query
- `selected_text`: Optional text selected by user for selected-text-only mode

**Validation rules**:
- Content must not be empty
- Conversation_id must reference an existing conversation
- Query_mode must match conversation mode when in selected-text-only mode

## Retrieved Passage Entity

**Description**: Book content retrieved by the system that is used to ground the response

**Fields**:
- `id`: Unique identifier for the passage
- `content`: The actual text content
- `source_document`: Reference to the source document (book/chapter)
- `chapter`: Chapter information
- `section`: Section information
- `page_number`: Page number in source
- `similarity_score`: Vector similarity score to the query
- `metadata`: Additional metadata (tags, content type, etc.)

**Validation rules**:
- Content must not be empty
- Source_document must be valid
- Similarity_score must be between 0 and 1

## Generated Response Entity

**Description**: The chatbot's answer generated based on retrieved passages

**Fields**:
- `id`: Unique identifier for the response
- `query_id`: Reference to the original query
- `content`: The generated response text
- `timestamp`: When the response was generated
- `source_passages`: Array of passage IDs used to generate the response
- `confidence_level`: Confidence score for the response
- `citations`: References to specific parts of the source passages

**Validation rules**:
- Content must be grounded in source_passages
- Query_id must reference an existing query
- Citations must reference actual parts of source passages

## User Selection Entity

**Description**: Specific text selected by the user for the selected-text-only query mode

**Fields**:
- `id`: Unique identifier for the selection
- `content`: The selected text content
- `context_before`: Context text before the selection (for positioning)
- `context_after`: Context text after the selection (for positioning)
- `source_document`: Reference to the source document
- `chapter`: Chapter information
- `section`: Section information
- `position_start`: Starting position in the source document
- `position_end`: Ending position in the source document

**Validation rules**:
- Content must not be empty
- Position_start must be less than position_end
- Source_document must be valid