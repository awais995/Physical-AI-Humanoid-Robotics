import asyncio
import logging
from typing import List, Union
import cohere
from ..config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using Cohere API
    """

    def __init__(self):
        self.co = cohere.Client(settings.cohere_api_key)
        self.model = "embed-multilingual-v3.0"  # Using multilingual model for broader language support
        self.batch_size = 96  # Cohere's max batch size for embeddings

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        """
        try:
            response = self.co.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"  # Using search_document for content retrieval
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts
        """
        all_embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self.co.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document"
                )
                all_embeddings.extend(response.embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//self.batch_size + 1}: {e}")
                raise

        return all_embeddings

    async def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query (optimized for search)
        """
        try:
            response = self.co.embed(
                texts=[query],
                model=self.model,
                input_type="search_query"  # Using search_query for better retrieval performance
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    async def rerank_documents(self, query: str, documents: List[str], top_n: int = 3) -> List[dict]:
        """
        Rerank documents based on relevance to query using Cohere's rerank API
        """
        try:
            response = self.co.rerank(
                model="rerank-english-v3.0",  # Updated to supported model
                query=query,
                documents=documents,
                top_n=top_n
            )
            return [
                {
                    "index": item.index,
                    "document": item.document,
                    "relevance_score": item.relevance_score
                }
                for item in response.results
            ]
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            # If reranking fails, return documents in original order with low scores
            return [
                {
                    "index": i,
                    "document": doc,
                    "relevance_score": 0.0
                }
                for i, doc in enumerate(documents[:top_n])
            ]

    async def get_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using their embeddings
        """
        try:
            emb1 = await self.get_embedding(text1)
            emb2 = await self.get_embedding(text2)

            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            magnitude1 = sum(a * a for a in emb1) ** 0.5
            magnitude2 = sum(a * a for a in emb2) ** 0.5

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            similarity = dot_product / (magnitude1 * magnitude2)
            return max(0.0, min(1.0, similarity))  # Clamp between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0


# Global embedding service instance
embedding_service = EmbeddingService()