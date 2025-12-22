import asyncio
import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..models.retrieved_passage import RetrievedPassage
from ..models.query import QueryMode
from ..config.settings import settings
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant passages from the vector database
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            https=True if "qdrant.io" in settings.qdrant_url else False
        )
        self.collection_name = settings.qdrant_collection_name
        self.top_k = settings.top_k
        self.similarity_threshold = settings.similarity_threshold
        self.rerank_enabled = settings.rerank_enabled
        self.rerank_top_k = settings.rerank_top_k

    async def retrieve_passages(self, query_text: str, query_mode: QueryMode = QueryMode.GLOBAL,
                               selected_text: Optional[str] = None, filters: Optional[Dict] = None) -> List[RetrievedPassage]:
        """
        Retrieve relevant passages based on query text and mode
        """
        try:
            # Get query embedding
            query_embedding = await self.embedding_service.get_query_embedding(query_text)

            # Build search filters based on query mode and provided filters
            search_filter = self._build_search_filter(query_mode, selected_text, filters)

            # Perform vector search - using method compatible with deployed Qdrant version
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=self.top_k,
                score_threshold=self.similarity_threshold
            )

            # Convert search results to RetrievedPassage objects
            passages = []
            for result in search_results:
                if result.score >= self.similarity_threshold:
                    payload = result.payload
                    passage = RetrievedPassage(
                        id=result.id,
                        content=payload.get("content", ""),
                        source_url=payload.get("source_url", ""),
                        source_title=payload.get("source_title", ""),
                        source_section=payload.get("source_section"),
                        context_before=payload.get("context_before"),
                        context_after=payload.get("context_after"),
                        similarity_score=result.score,
                        char_start_pos=payload.get("char_start_pos", 0),
                        char_end_pos=payload.get("char_end_pos", 0),
                        passage_metadata=payload.get("metadata", {})
                    )
                    passages.append(passage)

            # Apply reranking if enabled and we have passages
            if self.rerank_enabled and passages:
                passages = await self._rerank_passages(query_text, passages)

            logger.info(f"Retrieved {len(passages)} passages for query: {query_text[:50]}...")
            return passages

        except Exception as e:
            logger.error(f"Error retrieving passages: {e}")
            return []

    def _build_search_filter(self, query_mode: QueryMode, selected_text: Optional[str],
                           filters: Optional[Dict]) -> Optional[models.Filter]:
        """
        Build Qdrant search filter based on query mode and filters
        """
        conditions = []

        # If in selected-text-only mode, we still need to search in the database
        # but we'll handle the constraint in the generation service
        if filters:
            # Add custom filters (like specific chapters, sections, etc.)
            for key, value in filters.items():
                if key == "source_section":
                    conditions.append(
                        models.FieldCondition(
                            key="source_section",
                            match=models.MatchValue(value=value)
                        )
                    )
                elif key == "source_title":
                    conditions.append(
                        models.FieldCondition(
                            key="source_title",
                            match=models.MatchValue(value=value)
                        )
                    )

        if conditions:
            return models.Filter(must=conditions)
        else:
            return None

    async def _rerank_passages(self, query_text: str, passages: List[RetrievedPassage]) -> List[RetrievedPassage]:
        """
        Rerank passages based on relevance to query using Cohere's rerank API
        """
        try:
            # Extract content from passages for reranking
            documents = [passage.content for passage in passages]

            # Perform reranking
            reranked_results = await self.embedding_service.rerank_documents(
                query=query_text,
                documents=documents,
                top_n=self.rerank_top_k
            )

            # Reorder passages based on reranked results
            reranked_passages = []
            for result in reranked_results:
                original_index = result["index"]
                if original_index < len(passages):
                    # Update the similarity score with the reranked score
                    passage = passages[original_index]
                    updated_passage = RetrievedPassage(
                        id=passage.id,
                        content=passage.content,
                        source_url=passage.source_url,
                        source_title=passage.source_title,
                        source_section=passage.source_section,
                        context_before=passage.context_before,
                        context_after=passage.context_after,
                        similarity_score=result["relevance_score"],  # Use reranked score
                        char_start_pos=passage.char_start_pos,
                        char_end_pos=passage.char_end_pos,
                        passage_metadata=passage.passage_metadata
                    )
                    reranked_passages.append(updated_passage)

            logger.info(f"Reranked {len(passages)} passages to {len(reranked_passages)} top results")
            return reranked_passages

        except Exception as e:
            logger.error(f"Error reranking passages: {e}")
            # Return original passages if reranking fails
            return passages

    async def retrieve_passages_for_selected_text(self, query_text: str, selected_text: str) -> List[RetrievedPassage]:
        """
        Retrieve passages specifically for selected text only mode
        In this mode, we create a virtual passage from the selected text
        and potentially find similar passages to provide context
        """
        try:
            # First, check if the query is related to the selected text
            # Calculate similarity between query and selected text
            query_relevance = await self.embedding_service.get_similarity_score(query_text, selected_text)

            if query_relevance < self.similarity_threshold:
                # The query might not be directly related to the selected text
                # In selected-text-only mode, we should still provide relevant info
                # based on the selected text and query combination
                pass

            # Create a virtual passage from the selected text
            virtual_passage = RetrievedPassage(
                content=selected_text,
                source_url="selected_text_context",
                source_title="Selected Text Context",
                source_section="Selected Text",
                similarity_score=query_relevance,
                passage_metadata={
                    "type": "selected_text",
                    "original_query_relevance": query_relevance
                }
            )

            # Also retrieve similar passages from the database for additional context
            similar_passages = await self.retrieve_passages(
                f"{query_text} {selected_text}",  # Search for query + selected text context
                QueryMode.GLOBAL,
                filters=None
            )

            # Filter to only include passages that are highly relevant to the selected text
            context_passages = []
            for passage in similar_passages:
                passage_relevance = await self.embedding_service.get_similarity_score(selected_text, passage.content)
                if passage_relevance >= self.similarity_threshold:
                    # Update the passage with the relevance to selected text
                    updated_passage = RetrievedPassage(
                        id=passage.id,
                        content=passage.content,
                        source_url=passage.source_url,
                        source_title=passage.source_title,
                        source_section=passage.source_section,
                        context_before=passage.context_before,
                        context_after=passage.context_after,
                        similarity_score=passage_relevance,
                        char_start_pos=passage.char_start_pos,
                        char_end_pos=passage.char_end_pos,
                        passage_metadata=passage.passage_metadata
                    )
                    context_passages.append(updated_passage)

            # Combine the selected text as the primary passage with relevant context passages
            all_passages = [virtual_passage] + context_passages

            logger.info(f"Retrieved {len(all_passages)} passages for selected text query")
            return all_passages

        except Exception as e:
            logger.error(f"Error retrieving passages for selected text: {e}")
            # Return a passage with just the selected text if there's an error
            return [RetrievedPassage(
                content=selected_text,
                source_url="selected_text_context",
                source_title="Selected Text Context",
                source_section="Selected Text",
                similarity_score=1.0,
                passage_metadata={"type": "selected_text", "error_occurred": True}
            )]

    async def get_passage_by_id(self, passage_id: str) -> Optional[RetrievedPassage]:
        """
        Retrieve a specific passage by its ID
        """
        try:
            records = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[passage_id]
            )

            if records:
                record = records[0]
                payload = record.payload
                return RetrievedPassage(
                    id=record.id,
                    content=payload.get("content", ""),
                    source_url=payload.get("source_url", ""),
                    source_title=payload.get("source_title", ""),
                    source_section=payload.get("source_section"),
                    context_before=payload.get("context_before"),
                    context_after=payload.get("context_after"),
                    char_start_pos=payload.get("char_start_pos", 0),
                    char_end_pos=payload.get("char_end_pos", 0),
                    passage_metadata=payload.get("metadata", {})
                )

            return None
        except Exception as e:
            logger.error(f"Error retrieving passage by ID {passage_id}: {e}")
            return None


# Global retrieval service instance
retrieval_service = RetrievalService()