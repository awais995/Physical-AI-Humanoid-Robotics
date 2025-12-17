import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime
from ..models.retrieved_passage import RetrievedPassage
from ..config.settings import settings
from .embedding_service import EmbeddingService
from qdrant_client import QdrantClient
from qdrant_client.http import models
import re
import tiktoken

logger = logging.getLogger(__name__)


class ContentService:
    """
    Service for ingesting and processing book content for RAG system
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            https=True if "qdrant.io" in settings.qdrant_url else False
        )
        self.collection_name = settings.qdrant_collection_name
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use for token counting

    async def initialize_collection(self):
        """
        Initialize the Qdrant collection for storing book content
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create collection with appropriate vector size for Cohere embeddings
                # Cohere embeddings are 1024-dimensional
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def _tokenize_text(self, text: str) -> int:
        """
        Count the number of tokens in a text string
        """
        return len(self.enc.encode(text))

    def _split_text(self, text: str, source_url: str, source_title: str, source_section: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks of specified token size with overlap
        """
        chunks = []

        # Use a sliding window approach to chunk the text
        tokens = self.enc.encode(text)

        start_idx = 0
        while start_idx < len(tokens):
            # Determine the end index for this chunk
            end_idx = start_idx + self.chunk_size

            # If this is the last chunk, include all remaining tokens
            if end_idx > len(tokens):
                end_idx = len(tokens)

            # Decode the chunk back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.enc.decode(chunk_tokens)

            # Add context before and after
            context_before = ""
            context_after = ""

            # Add context from overlapping tokens if available
            context_start = max(0, start_idx - self.chunk_overlap)
            context_end = min(len(tokens), end_idx + self.chunk_overlap)

            if context_start < start_idx:
                context_tokens = tokens[context_start:start_idx]
                context_before = self.enc.decode(context_tokens)

            if end_idx < context_end:
                context_tokens = tokens[end_idx:context_end]
                context_after = self.enc.decode(context_tokens)

            # Create a chunk with metadata
            chunk = {
                "content": chunk_text,
                "source_url": source_url,
                "source_title": source_title,
                "source_section": source_section,
                "context_before": context_before,
                "context_after": context_after,
                "char_start_pos": len(self.enc.decode(tokens[:start_idx])),
                "char_end_pos": len(self.enc.decode(tokens[:end_idx])),
                "token_count": len(chunk_tokens)
            }

            chunks.append(chunk)

            # Move to the next chunk position (with overlap)
            start_idx = end_idx - (self.chunk_overlap if end_idx < len(tokens) else 0)

        return chunks

    async def _create_passage_from_chunk(self, chunk: Dict[str, Any], idx: int) -> RetrievedPassage:
        """
        Create a RetrievedPassage object from a chunk of text
        """
        # Create a unique ID as a proper UUID
        import uuid
        passage_id = str(uuid.uuid4())

        return RetrievedPassage(
            id=passage_id,
            content=chunk["content"],
            source_url=chunk["source_url"],
            source_title=chunk["source_title"],
            source_section=chunk["source_section"],
            context_before=chunk["context_before"],
            context_after=chunk["context_after"],
            char_start_pos=chunk["char_start_pos"],
            char_end_pos=chunk["char_end_pos"],
            passage_metadata={
                "chunk_index": idx,
                "token_count": chunk["token_count"],
                "created_at": datetime.now().isoformat()
            }
        )

    async def _embed_and_store_passage(self, passage: RetrievedPassage):
        """
        Generate embedding for passage and store in Qdrant
        """
        try:
            # Generate embedding using Cohere
            embedding = await self.embedding_service.get_embedding(passage.content)

            # Prepare payload for Qdrant
            payload = {
                "content": passage.content,
                "source_url": passage.source_url,
                "source_title": passage.source_title,
                "source_section": passage.source_section,
                "context_before": passage.context_before,
                "context_after": passage.context_after,
                "char_start_pos": passage.char_start_pos,
                "char_end_pos": passage.char_end_pos,
                "metadata": passage.passage_metadata
            }

            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=passage.id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            logger.debug(f"Stored passage {passage.id} in Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error embedding and storing passage {passage.id}: {e}")
            return False

    async def ingest_document(self, content: str, source_url: str, source_title: str, source_section: Optional[str] = None):
        """
        Ingest a single document and store it in the vector database
        """
        try:
            logger.info(f"Starting ingestion for document: {source_title}")

            # Split the content into chunks
            chunks = self._split_text(content, source_url, source_title, source_section)
            logger.info(f"Split document into {len(chunks)} chunks")

            # Process each chunk
            successful_chunks = 0
            for idx, chunk in enumerate(chunks):
                # Create passage from chunk
                passage = await self._create_passage_from_chunk(chunk, idx)

                # Embed and store the passage
                success = await self._embed_and_store_passage(passage)

                if success:
                    successful_chunks += 1
                else:
                    logger.warning(f"Failed to store chunk {idx}")

                # Add small delay to avoid overwhelming the API
                await asyncio.sleep(0.01)

            logger.info(f"Successfully ingested {successful_chunks}/{len(chunks)} chunks for {source_title}")
            return successful_chunks

        except Exception as e:
            logger.error(f"Error ingesting document {source_title}: {e}")
            raise

    async def ingest_documents_batch(self, documents: List[Dict[str, str]]):
        """
        Ingest multiple documents in batch
        """
        try:
            logger.info(f"Starting batch ingestion of {len(documents)} documents")

            total_successful = 0
            for doc in documents:
                content = doc.get("content", "")
                source_url = doc.get("source_url", "")
                source_title = doc.get("source_title", "")
                source_section = doc.get("source_section")

                if not content or not source_url or not source_title:
                    logger.warning(f"Skipping document with missing required fields: {source_title}")
                    continue

                successful = await self.ingest_document(content, source_url, source_title, source_section)
                total_successful += successful

            logger.info(f"Batch ingestion completed. Total chunks ingested: {total_successful}")
            return total_successful

        except Exception as e:
            logger.error(f"Error in batch ingestion: {e}")
            raise

    async def get_content_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed content
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            # Get a sample of passages to understand content diversity
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=5  # Get 5 sample passages
            )

            stats = {
                "total_passages": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_type": collection_info.config.params.vectors.distance,
                "sample_passages": []
            }

            for point in scroll_result[0]:
                stats["sample_passages"].append({
                    "id": point.id,
                    "source_title": point.payload.get("source_title"),
                    "source_section": point.payload.get("source_section"),
                    "content_preview": point.payload.get("content", "")[:100] + "..."
                })

            return stats

        except Exception as e:
            logger.error(f"Error getting content stats: {e}")
            return {"error": str(e)}

    async def clear_collection(self):
        """
        Clear all content from the collection (useful for re-indexing)
        """
        try:
            # Delete and recreate the collection to clear all content
            self.qdrant_client.delete_collection(self.collection_name)
            await self.initialize_collection()
            logger.info(f"Cleared and recreated collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


# Global content service instance
content_service = ContentService()