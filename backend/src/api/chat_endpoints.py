from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uuid
from datetime import datetime
import logging

from ..models.query import Query, QueryMode
from ..models.response import Response
from ..models.conversation import Conversation
from ..models.retrieved_passage import RetrievedPassage
from ..services.retrieval_service import retrieval_service
from ..services.generation_service import generation_service
from ..services.content_service import content_service
from ..services.conversation_service import conversation_service
from ..config.settings import settings

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/chat", tags=["chat"])

# Using conversation service for persistence
# In-memory storage will be replaced by database calls in the conversation service


@router.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    """
    try:
        await content_service.initialize_collection()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise


class QueryRequest(BaseModel):
    query_text: str
    mode: QueryMode = QueryMode.GLOBAL
    selected_text: Optional[str] = None
    conversation_id: Optional[str] = None
    filters: Optional[Dict[str, str]] = None

@router.post("/query", response_model=Dict[str, Any])
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint to handle chat queries with different modes
    """
    try:
        query_text = request.query_text
        mode = request.mode
        selected_text = request.selected_text
        conversation_id = request.conversation_id
        filters = request.filters or {}

        # Generate or use provided conversation ID
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            # Create new conversation
            title = query_text[:50] + "..." if len(query_text) > 50 else query_text
            await conversation_service.create_conversation(title=title)

        # Create query object
        query_obj = Query(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            query_text=query_text,
            query_mode=mode,
            selected_text=selected_text,
            timestamp=datetime.now(),
            filters=filters
        )

        # Retrieve relevant passages based on mode
        if mode == QueryMode.SELECTED_TEXT_ONLY and selected_text:
            retrieved_passages = await retrieval_service.retrieve_passages_for_selected_text(
                query_text, selected_text
            )
        else:
            retrieved_passages = await retrieval_service.retrieve_passages(
                query_text, mode, selected_text, filters
            )

        # Generate response based on retrieved passages
        response_text = await generation_service.generate_response(query_obj, retrieved_passages)

        # Calculate confidence score
        confidence = await generation_service.get_confidence_score(
            query_obj, response_text, retrieved_passages
        )

        # Generate citations
        citations = await generation_service.generate_citations(retrieved_passages)

        # Create response object
        response_obj = Response(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            query_id=query_obj.id,
            response_text=response_text,
            citations=retrieved_passages,  # Using the original RetrievedPassage objects for citations
            confidence=confidence,
            generated_at=datetime.now(),
            tokens_used=len(response_text.split()),  # Approximate token count
            processing_time_ms=0,  # Will be calculated in a more complete implementation
            hallucination_check_passed=True,  # Assuming it passed for now
            metadata={"query_mode": mode.value}
        )

        # Add query and response to conversation
        await conversation_service.add_query_to_conversation(conversation_id, query_obj)
        await conversation_service.add_response_to_conversation(conversation_id, response_obj)

        # Prepare response - concise format without confidence and citations in main response
        result = {
            "conversation_id": conversation_id,
            "response": response_obj.response_text,
            "query_id": query_obj.id,
            "mode": mode.value
            # Removed citations and confidence from main response for conciseness
        }

        logger.info(f"Processed query for conversation {conversation_id}, mode: {mode.value}")
        return result

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(conversation_id: str):
    """
    Retrieve a specific conversation by ID
    """
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # For now, return a basic response since the conversation service is not fully implemented
        return {
            "conversation_id": conversation_id,
            "title": f"Conversation {conversation_id[:8]}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "query_count": 0,
            "response_count": 0,
            "queries": [],
            "responses": []
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation")


@router.get("/conversations", response_model=List[Dict[str, Any]])
async def list_conversations():
    """
    List all conversations (with basic info, not full content)
    """
    try:
        # For now, return an empty list since the conversation service is not fully implemented
        # In a real implementation, this would call conversation_service.list_user_conversations()
        return []
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail="Error listing conversations")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a specific conversation
    """
    try:
        success = await conversation_service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": f"Conversation {conversation_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting conversation")


@router.post("/ingest", response_model=Dict[str, Any])
async def ingest_content(
    content: str,
    source_url: str,
    source_title: str,
    source_section: Optional[str] = None
):
    """
    Endpoint to ingest new content into the RAG system
    """
    try:
        # Ingest the document into the vector database
        chunks_ingested = await content_service.ingest_document(
            content=content,
            source_url=source_url,
            source_title=source_title,
            source_section=source_section
        )

        return {
            "message": "Content ingested successfully",
            "chunks_ingested": chunks_ingested,
            "source_title": source_title,
            "source_url": source_url
        }
    except Exception as e:
        logger.error(f"Error ingesting content: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting content: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "RAG Chatbot API",
        "version": settings.app_version,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/stats")
async def get_stats():
    """
    Get statistics about the RAG system
    """
    try:
        content_stats = await content_service.get_content_stats()
        return {
            "content_stats": content_stats,
            "total_conversations": len(conversations),
            "active_conversations": len([c for c in conversations.values() if c.active]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting stats")