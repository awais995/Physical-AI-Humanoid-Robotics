# Quickstart Guide: Integrated RAG Chatbot for the Docusaurus Book

## Overview

This guide provides instructions for setting up and using the Retrieval-Augmented Generation (RAG) chatbot that integrates with the Docusaurus book site. The system provides two query modes: global book retrieval and selected-text-only retrieval, with responses grounded 100% in book content.

## Prerequisites

- Python 3.11+
- Node.js 16+ (for frontend development)
- Cohere API key
- Qdrant Cloud account
- Neon Postgres account

## Backend Setup

### 1. Environment Configuration

Create a `.env` file in the backend directory with the following variables:

```bash
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
DATABASE_URL=your_neon_postgres_connection_string
COHERE_EMBED_MODEL=embed-english-v3.0
COHERE_GENERATE_MODEL=command-r-plus
```

### 2. Installation

```bash
cd backend
pip install -r requirements.txt
```

### 3. Database Setup

The system uses Neon Postgres for conversation storage and Qdrant Cloud for vector embeddings. Ensure your database credentials are properly configured in the environment variables.

## Content Ingestion

Run the content ingestion script to process the book content:
```bash
python -m src.ingestion.ingest_content --source-path path/to/book/content
```

This will chunk the content and store embeddings in Qdrant Cloud.

## Running the System

### Backend (FastAPI)
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the backend server:
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Frontend Integration
1. The chatbot UI is integrated directly into the Docusaurus site
2. The widget appears as a floating button on book pages
3. Users can select text and ask questions about it

## Frontend Integration

### 1. Environment Configuration

In your Docusaurus project, add the following to your environment:

```bash
REACT_APP_API_BASE_URL=http://localhost:8000  # or your backend URL
```

### 2. Component Integration

The chatbot components are located in `frontend/src/components/RagChatbot/`:

- `ChatInterface.jsx` - Main chat interface
- `QueryModeSelector.jsx` - Mode selection (global vs selected-text-only)
- `ResponseRenderer.jsx` - Response display with citations
- `ConversationHistory.jsx` - Conversation history display

## API Endpoints

### Chat Query
```
POST /chat/query
```

Request body:
```json
{
  "query": "Your question here",
  "mode": "global" | "selected-text-only",
  "selected_text": "Text selected by user (required for selected-text-only mode)",
  "conversation_id": "Optional conversation ID for continuing a conversation"
}
```

Response:
```json
{
  "response": "Generated response",
  "citations": [{"source": "...", "text": "...", "chapter": "...", "section": "..."}],
  "conversation_id": "ID of the conversation",
  "confidence": 0.8,
  "mode": "global"
}
```

### Get Conversation History
```
GET /chat/conversation/{conversation_id}
```

### List Conversations
```
GET /chat/conversations
```

## Usage Modes

### 1. Global Book Retrieval (Default)

Queries the entire book content for relevant information:

```javascript
const response = await chatAPI.queryChat({
  query: "Explain artificial intelligence in robotics",
  mode: "global"
});
```

### 2. Selected Text Only

Only answers questions based on user-selected text:

```javascript
const response = await chatAPI.queryChat({
  query: "What does this text say about neural networks?",
  mode: "selected-text-only",
  selected_text: "User selected text content here..."
});
```

## Query Modes

### Global Book Retrieval
- Accesses the entire book content
- Best for general questions
- Activated by default

### Selected-Text-Only Mode
- Restricts responses to user-selected text
- Strictly enforces the constraint (no fallback)
- Activated when user selects text and uses the chat interface

## Caching Strategy

The system implements caching for identical queries to improve performance and reduce API costs. Query results are cached for 30 minutes by default.

## Rate Limiting

Free-tier constraints are enforced with rate limiting:
- Query endpoint: 10 requests per minute per IP
- History endpoint: 30 requests per minute per IP
- List endpoint: 20 requests per minute per IP

## Conversation Persistence

Conversations are stored in Neon Postgres with:
- Conversation metadata (creation time, mode, session token)
- Query and response history
- Citations and confidence scores

## Validation and Grounding

The system ensures all responses are 100% grounded in book content:
- Content validation using semantic similarity
- Citation alignment verification
- Selected-text-only mode enforcement
- Confidence scoring based on passage relevance

## Testing

Run backend tests:
```bash
cd backend
pytest tests/
```

## Deployment

1. Set up production environment variables
2. Deploy backend to a cloud provider (ensure it has access to Cohere, Qdrant, and Neon)
3. Integrate frontend components into your Docusaurus build process
4. Configure CORS settings appropriately

## Troubleshooting

- If queries return "Cannot answer from provided text", ensure the selected text contains relevant information
- If responses seem unrelated, check the similarity threshold settings
- For performance issues, verify your Cohere, Qdrant, and Neon connection speeds
- For caching issues, the system will automatically clean up expired cache entries
- If queries return "I couldn't find relevant information", ensure your content has been properly ingested into Qdrant

## Performance Goals

- Response time under 5 seconds for 95% of requests (SC-005)
- 95% refusal rate for selected-text-only mode when information is not available (SC-002)
- 90% accuracy in providing relevant answers (SC-004)
- No breaking of existing Docusaurus functionality (SC-006)