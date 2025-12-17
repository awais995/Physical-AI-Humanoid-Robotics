import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.config.settings import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for tests to avoid requiring real API keys"""
    with patch('src.config.settings.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-cohere-key"
        mock_settings.qdrant_url = "http://localhost:6333"
        mock_settings.qdrant_api_key = "test-qdrant-key"
        mock_settings.neon_database_url = "postgresql://test:test@localhost/test"
        mock_settings.app_version = "test-1.0.0"
        mock_settings.debug = True
        mock_settings.top_k = 5
        mock_settings.similarity_threshold = 0.3
        mock_settings.chunk_size = 512
        mock_settings.chunk_overlap = 50

        yield mock_settings


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing"""
    with patch('src.services.content_service.QdrantClient') as mock_class:
        mock_instance = MagicMock()
        mock_instance.get_collections.return_value = MagicMock(collections=[])
        mock_instance.create_collection.return_value = None
        mock_instance.upsert.return_value = None
        mock_instance.search.return_value = []
        mock_instance.get_collection.return_value = MagicMock(
            points_count=100,
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=1024, distance="cosine")))
        )
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_cohere_client():
    """Mock Cohere client for testing"""
    with patch('src.services.embedding_service.cohere.Client') as mock_class:
        mock_instance = MagicMock()
        mock_instance.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3]])
        mock_instance.generate.return_value = MagicMock(
            generations=[MagicMock(text="Test response")]
        )
        mock_instance.rerank.return_value = MagicMock(results=[])
        mock_class.return_value = mock_instance
        yield mock_instance