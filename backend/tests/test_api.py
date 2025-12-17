import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from src.main import app
from src.services.retrieval_service import retrieval_service
from src.services.generation_service import generation_service
from src.models.query import Query, QueryMode

client = TestClient(app)


@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service for testing"""
    with patch('src.api.chat_endpoints.retrieval_service') as mock:
        yield mock


@pytest.fixture
def mock_generation_service():
    """Mock generation service for testing"""
    with patch('src.api.chat_endpoints.generation_service') as mock:
        yield mock


@pytest.fixture
def mock_conversation_service():
    """Mock conversation service for testing"""
    with patch('src.api.chat_endpoints.conversation_service') as mock:
        yield mock


class TestChatEndpoints:
    """Test cases for the chat API endpoints"""

    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/chat/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_query_endpoint_global_mode(self, mock_retrieval_service, mock_generation_service, mock_conversation_service):
        """Test the query endpoint with global mode"""
        # Mock the services
        mock_retrieval_service.retrieve_passages.return_value = []
        mock_generation_service.generate_response.return_value = "This is a test response"
        mock_generation_service.get_confidence_score.return_value = 0.9
        mock_generation_service.generate_citations.return_value = []

        mock_conversation_service.create_conversation.return_value = AsyncMock()
        mock_conversation_service.add_query_to_conversation.return_value = AsyncMock()
        mock_conversation_service.add_response_to_conversation.return_value = AsyncMock()

        # Make request
        payload = {
            "query_text": "What is the book about?",
            "mode": "global",
            "selected_text": None,
            "conversation_id": None
        }

        response = client.post("/chat/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "This is a test response"
        assert "confidence" in data

    def test_query_endpoint_selected_text_mode(self, mock_retrieval_service, mock_generation_service, mock_conversation_service):
        """Test the query endpoint with selected text mode"""
        # Mock the services
        mock_retrieval_service.retrieve_passages_for_selected_text.return_value = []
        mock_generation_service.generate_response.return_value = "This is a response based on selected text"
        mock_generation_service.get_confidence_score.return_value = 0.85
        mock_generation_service.generate_citations.return_value = []

        mock_conversation_service.create_conversation.return_value = AsyncMock()
        mock_conversation_service.add_query_to_conversation.return_value = AsyncMock()
        mock_conversation_service.add_response_to_conversation.return_value = AsyncMock()

        # Make request
        payload = {
            "query_text": "Explain this concept?",
            "mode": "selected_text_only",
            "selected_text": "This is the selected text that the user has highlighted.",
            "conversation_id": None
        }

        response = client.post("/chat/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "This is a response based on selected text"
        assert "confidence" in data

    def test_query_endpoint_missing_text(self):
        """Test query endpoint with missing query text"""
        payload = {
            "query_text": "",
            "mode": "global",
            "selected_text": None,
            "conversation_id": None
        }

        response = client.post("/chat/query", json=payload)
        # This should fail since query_text is empty
        assert response.status_code == 500  # or appropriate error status

    def test_get_conversation(self, mock_conversation_service):
        """Test getting a specific conversation"""
        # Mock the service
        mock_conversation = MagicMock()
        mock_conversation.id = "test-conversation-id"
        mock_conversation_service.get_conversation.return_value = mock_conversation

        response = client.get("/chat/conversations/test-conversation-id")

        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data

    def test_list_conversations(self):
        """Test listing all conversations"""
        response = client.get("/chat/conversations")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_delete_conversation(self, mock_conversation_service):
        """Test deleting a conversation"""
        mock_conversation_service.delete_conversation.return_value = True

        response = client.delete("/chat/conversations/test-conversation-id")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_ingest_content(self, mock_conversation_service):
        """Test ingesting new content"""
        payload = {
            "content": "This is test content for ingestion.",
            "source_url": "http://example.com/test",
            "source_title": "Test Content",
            "source_section": "Chapter 1"
        }

        response = client.post("/chat/ingest", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "chunks_ingested" in data


class TestServices:
    """Test cases for the backend services"""

    @pytest.mark.asyncio
    async def test_retrieval_service(self):
        """Test retrieval service functionality"""
        # This would require actual Qdrant connection, so we'll mock it
        with patch.object(retrieval_service.qdrant_client, 'search') as mock_search:
            mock_search.return_value = []

            passages = await retrieval_service.retrieve_passages("test query")
            assert isinstance(passages, list)

    @pytest.mark.asyncio
    async def test_generation_service(self):
        """Test generation service functionality"""
        # This would require actual Cohere connection, so we'll test the structure
        query = Query(
            id="test-id",
            conversation_id="test-conv-id",
            query_text="test query",
            query_mode=QueryMode.GLOBAL,
            timestamp=MagicMock()
        )

        with patch.object(generation_service.co, 'generate') as mock_generate:
            mock_generate.return_value.generations = [MagicMock(text="Test response")]

            response = await generation_service.generate_response(query, [])
            assert response == "Test response"


if __name__ == "__main__":
    pytest.main()