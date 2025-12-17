import pytest
from datetime import datetime
from src.models.conversation import Conversation
from src.models.query import Query, QueryMode
from src.models.response import Response
from src.models.retrieved_passage import RetrievedPassage


class TestConversationModel:
    """Test cases for the Conversation model"""

    def test_conversation_creation(self):
        """Test creating a conversation"""
        conversation = Conversation(
            id="test-id",
            user_id="user-123",
            title="Test Conversation",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            active=True
        )

        assert conversation.id == "test-id"
        assert conversation.user_id == "user-123"
        assert conversation.title == "Test Conversation"
        assert conversation.active is True
        assert len(conversation.queries) == 0
        assert len(conversation.responses) == 0

    def test_conversation_defaults(self):
        """Test conversation with default values"""
        conversation = Conversation(
            title="Test Conversation",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        assert conversation.id is None  # Should be None until assigned
        assert conversation.user_id is None
        assert conversation.title == "Test Conversation"
        assert conversation.active is True


class TestQueryModel:
    """Test cases for the Query model"""

    def test_query_creation(self):
        """Test creating a query"""
        query = Query(
            id="query-123",
            conversation_id="conv-456",
            query_text="What is this about?",
            query_mode=QueryMode.GLOBAL,
            selected_text=None,
            timestamp=datetime.now()
        )

        assert query.id == "query-123"
        assert query.conversation_id == "conv-456"
        assert query.query_text == "What is this about?"
        assert query.query_mode == QueryMode.GLOBAL
        assert query.selected_text is None
        assert query.timestamp is not None

    def test_query_selected_text_mode(self):
        """Test query with selected text mode"""
        selected_text = "This is the selected text."
        query = Query(
            conversation_id="conv-456",
            query_text="Explain this?",
            query_mode=QueryMode.SELECTED_TEXT_ONLY,
            selected_text=selected_text,
            timestamp=datetime.now()
        )

        assert query.query_mode == QueryMode.SELECTED_TEXT_ONLY
        assert query.selected_text == selected_text

    def test_query_defaults(self):
        """Test query with default values"""
        query = Query(
            conversation_id="conv-456",
            query_text="Test query",
            timestamp=datetime.now()
        )

        assert query.query_mode == QueryMode.GLOBAL  # Should default to GLOBAL
        assert query.filters == {}  # Should default to empty dict


class TestResponseModel:
    """Test cases for the Response model"""

    def test_response_creation(self):
        """Test creating a response"""
        response = Response(
            id="response-123",
            conversation_id="conv-456",
            query_id="query-789",
            response_text="This is the response.",
            citations=[],
            confidence=0.85,
            generated_at=datetime.now(),
            tokens_used=25,
            processing_time_ms=1250.5,
            hallucination_check_passed=True
        )

        assert response.id == "response-123"
        assert response.conversation_id == "conv-456"
        assert response.query_id == "query-789"
        assert response.response_text == "This is the response."
        assert response.citations == []
        assert response.confidence == 0.85
        assert response.tokens_used == 25
        assert response.processing_time_ms == 1250.5
        assert response.hallucination_check_passed is True

    def test_response_defaults(self):
        """Test response with default values"""
        response = Response(
            conversation_id="conv-456",
            query_id="query-789",
            response_text="Test response",
            generated_at=datetime.now()
        )

        assert response.confidence == 0.0  # Should default to 0.0
        assert response.tokens_used == 0  # Should default to 0
        assert response.processing_time_ms == 0.0  # Should default to 0.0
        assert response.hallucination_check_passed is True  # Should default to True


class TestRetrievedPassageModel:
    """Test cases for the RetrievedPassage model"""

    def test_retrieved_passage_creation(self):
        """Test creating a retrieved passage"""
        passage = RetrievedPassage(
            content="This is the passage content.",
            source_url="http://example.com",
            source_title="Example Title",
            source_section="Chapter 1",
            source_anchor="section-1",
            similarity_score=0.75,
            passage_metadata={"type": "paragraph", "page": 25}
        )

        assert passage.content == "This is the passage content."
        assert passage.source_url == "http://example.com"
        assert passage.source_title == "Example Title"
        assert passage.source_section == "Chapter 1"
        assert passage.source_anchor == "section-1"
        assert passage.similarity_score == 0.75
        assert passage.passage_metadata["type"] == "paragraph"
        assert passage.passage_metadata["page"] == 25

    def test_retrieved_passage_defaults(self):
        """Test retrieved passage with default values"""
        passage = RetrievedPassage(
            content="Test content",
            source_url="http://example.com",
            source_title="Test Title"
        )

        assert passage.similarity_score == 0.0  # Should default to 0.0
        assert passage.passage_metadata == {}  # Should default to empty dict
        assert passage.source_section is None  # Should default to None
        assert passage.source_anchor is None  # Should default to None
        assert len(passage.id) > 0  # Should have auto-generated ID


class TestModelRelationships:
    """Test relationships between models"""

    def test_conversation_query_response_relationship(self):
        """Test that conversations can contain queries and responses"""
        conv_time = datetime.now()
        conversation = Conversation(
            id="conv-123",
            title="Test Conversation",
            created_at=conv_time,
            updated_at=conv_time
        )

        # Create a query
        query = Query(
            id="query-1",
            conversation_id="conv-123",
            query_text="Test question?",
            timestamp=conv_time
        )

        # Create a response
        response = Response(
            id="response-1",
            conversation_id="conv-123",
            query_id="query-1",
            response_text="Test answer.",
            generated_at=conv_time
        )

        # Add them to the conversation
        conversation.queries.append(query)
        conversation.responses.append(response)

        assert len(conversation.queries) == 1
        assert len(conversation.responses) == 1
        assert conversation.queries[0].id == "query-1"
        assert conversation.responses[0].id == "response-1"

    def test_response_citations(self):
        """Test that responses can contain citations"""
        response = Response(
            conversation_id="conv-123",
            query_id="query-1",
            response_text="Test response with citations.",
            generated_at=datetime.now()
        )

        # Create some citations
        citation1 = RetrievedPassage(
            content="First citation content.",
            source_url="http://example1.com",
            source_title="Source 1"
        )

        citation2 = RetrievedPassage(
            content="Second citation content.",
            source_url="http://example2.com",
            source_title="Source 2"
        )

        # Add citations to response (in a real scenario, this would be done differently)
        # For testing purposes, we'll directly assign
        response.citations = [citation1, citation2]

        assert len(response.citations) == 2
        assert response.citations[0].source_title == "Source 1"
        assert response.citations[1].source_title == "Source 2"


if __name__ == "__main__":
    pytest.main()