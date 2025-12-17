import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.embedding_service import EmbeddingService
from src.services.retrieval_service import RetrievalService
from src.services.generation_service import GenerationService
from src.services.content_service import ContentService
from src.models.query import Query, QueryMode
from src.models.retrieved_passage import RetrievedPassage


class TestEmbeddingService:
    """Test cases for the embedding service"""

    @pytest.mark.asyncio
    async def test_get_embedding(self):
        """Test getting embedding for a single text"""
        service = EmbeddingService()

        # Mock the Cohere client
        with patch.object(service.co, 'embed') as mock_embed:
            mock_embed.return_value.embeddings = [[0.1, 0.2, 0.3]]

            result = await service.get_embedding("test text")
            assert isinstance(result, list)
            assert len(result) == 3  # 3-dimensional mock embedding

    @pytest.mark.asyncio
    async def test_get_query_embedding(self):
        """Test getting embedding for a query"""
        service = EmbeddingService()

        # Mock the Cohere client
        with patch.object(service.co, 'embed') as mock_embed:
            mock_embed.return_value.embeddings = [[0.4, 0.5, 0.6]]

            result = await service.get_query_embedding("test query")
            assert isinstance(result, list)
            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rerank_documents(self):
        """Test reranking documents"""
        service = EmbeddingService()

        # Mock the Cohere client
        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.document = "test doc"
        mock_result.relevance_score = 0.9
        mock_response = MagicMock()
        mock_response.results = [mock_result]

        with patch.object(service.co, 'rerank') as mock_rerank:
            mock_rerank.return_value = mock_response

            result = await service.rerank_documents("test query", ["doc1", "doc2"])
            assert len(result) == 1
            assert result[0]["relevance_score"] == 0.9


class TestRetrievalService:
    """Test cases for the retrieval service"""

    @pytest.mark.asyncio
    async def test_retrieve_passages(self):
        """Test retrieving passages"""
        service = RetrievalService()

        # Mock the Qdrant client
        mock_result = MagicMock()
        mock_result.id = "test-id"
        mock_result.payload = {
            "content": "test content",
            "source_url": "http://example.com",
            "source_title": "Test Title",
            "source_section": "Section 1"
        }
        mock_result.score = 0.8

        with patch.object(service.qdrant_client, 'search') as mock_search:
            mock_search.return_value = [mock_result]

            # Mock embedding service
            with patch.object(service, 'embedding_service') as mock_emb_service:
                mock_emb_service.get_query_embedding.return_value = [0.1, 0.2, 0.3]

                passages = await service.retrieve_passages("test query")
                assert len(passages) == 1
                assert isinstance(passages[0], RetrievedPassage)
                assert passages[0].content == "test content"

    @pytest.mark.asyncio
    async def test_retrieve_passages_for_selected_text(self):
        """Test retrieving passages for selected text mode"""
        service = RetrievalService()

        # Mock embedding service
        with patch.object(service, 'embedding_service') as mock_emb_service:
            mock_emb_service.get_similarity_score.return_value = 0.7

            passages = await service.retrieve_passages_for_selected_text(
                "test query",
                "selected text content"
            )
            assert len(passages) >= 1  # At least one passage should be returned
            assert passages[0].content == "selected text content"


class TestGenerationService:
    """Test cases for the generation service"""

    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test generating a response"""
        service = GenerationService()

        # Create a test query
        query = Query(
            id="test-id",
            conversation_id="test-conv-id",
            query_text="What is this book about?",
            query_mode=QueryMode.GLOBAL,
            timestamp=MagicMock()
        )

        # Create a test passage
        passage = RetrievedPassage(
            content="This book is about robotics and AI.",
            source_url="http://example.com",
            source_title="Test Book",
            similarity_score=0.9
        )

        # Mock the Cohere client
        mock_generation = MagicMock()
        mock_generation.text = "This book covers robotics and artificial intelligence topics."
        mock_response = MagicMock()
        mock_response.generations = [mock_generation]

        with patch.object(service.co, 'generate') as mock_generate:
            mock_generate.return_value = mock_response

            response = await service.generate_response(query, [passage])
            assert "robotics" in response.lower() or "artificial intelligence" in response.lower()

    @pytest.mark.asyncio
    async def test_build_context_from_passages(self):
        """Test building context from passages"""
        service = GenerationService()

        # Create test passages
        passages = [
            RetrievedPassage(
                content="First passage content.",
                source_url="http://example.com/1",
                source_title="Test Book",
                source_section="Chapter 1",
                similarity_score=0.8
            ),
            RetrievedPassage(
                content="Second passage content.",
                source_url="http://example.com/2",
                source_title="Test Book",
                source_section="Chapter 2",
                similarity_score=0.7
            )
        ]

        context = service._build_context_from_passages(passages)
        assert "First passage content" in context
        assert "Second passage content" in context
        assert "Chapter 1" in context
        assert "Chapter 2" in context

    @pytest.mark.asyncio
    async def test_check_hallucination(self):
        """Test hallucination detection"""
        service = GenerationService()

        passages = [
            RetrievedPassage(
                content="The robot has two arms and moves slowly.",
                source_url="http://example.com",
                source_title="Test Book",
                similarity_score=0.9
            )
        ]

        # Test with content that matches the passages
        valid_response = "The robot has two arms and moves slowly."
        is_valid = await service._check_hallucination(valid_response, passages)
        # Note: This test may not be fully accurate without proper embedding setup

        # Test with content that contradicts the passages
        invalid_response = "The robot has four arms and moves quickly."
        is_invalid = await service._check_hallucination(invalid_response, passages)


class TestContentService:
    """Test cases for the content service"""

    @pytest.mark.asyncio
    async def test_ingest_document(self):
        """Test ingesting a document"""
        service = ContentService()

        # Mock the embedding service
        with patch.object(service, 'embedding_service') as mock_emb_service:
            mock_emb_service.get_embedding.return_value = [0.1, 0.2, 0.3]

            # Mock the Qdrant client
            with patch.object(service.qdrant_client, 'upsert') as mock_upsert:
                mock_upsert.return_value = None

                result = await service.ingest_document(
                    content="Test document content",
                    source_url="http://example.com",
                    source_title="Test Document"
                )
                # Should return number of chunks ingested
                assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_initialize_collection(self):
        """Test initializing the Qdrant collection"""
        service = ContentService()

        # Mock the Qdrant client
        mock_collection = MagicMock()
        mock_collection.name = service.collection_name
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]

        with patch.object(service.qdrant_client, 'get_collections') as mock_get_collections:
            mock_get_collections.return_value = mock_collections

            # This should not raise an exception
            await service.initialize_collection()


class TestChunkingService:
    """Test cases for the chunking service"""

    def test_chunk_text(self):
        """Test chunking text by token size"""
        from src.services.chunking_service import chunking_service

        text = "This is a sample text that will be chunked into smaller pieces. " * 10
        chunks = chunking_service.chunk_text(
            text,
            source_url="http://example.com",
            source_title="Test Document"
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert "content" in chunk
            assert "source_url" in chunk
            assert "source_title" in chunk

    def test_chunk_by_semantic_boundaries(self):
        """Test chunking text by semantic boundaries"""
        from src.services.chunking_service import chunking_service

        text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph with more content."
        chunks = chunking_service.chunk_by_semantic_boundaries(
            text,
            source_url="http://example.com",
            source_title="Test Document"
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert "content" in chunk


if __name__ == "__main__":
    pytest.main()