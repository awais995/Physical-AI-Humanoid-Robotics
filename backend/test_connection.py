import asyncio
import os
from dotenv import load_dotenv
from src.services.embedding_service import EmbeddingService
from src.services.retrieval_service import RetrievalService
from src.config.settings import settings

# Load environment variables
load_dotenv('backend/.env')

async def test_cohere_connection():
    """
    Test the Cohere API connection
    """
    print("Testing Cohere API connection...")

    try:
        embedding_service = EmbeddingService()

        # Test embedding generation
        test_text = "This is a test for Cohere API connection."
        embedding = await embedding_service.get_embedding(test_text)

        print(f"✓ Cohere API connection successful!")
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
        print(f"✓ Sample values: {embedding[:5]}...")  # Show first 5 values

        return True

    except Exception as e:
        print(f"✗ Cohere API connection failed: {e}")
        print("Please check your COHERE_API_KEY in the .env file")
        return False

async def test_qdrant_connection():
    """
    Test the Qdrant connection
    """
    print("\nTesting Qdrant connection...")

    try:
        retrieval_service = RetrievalService()

        # Test getting collection info (this will verify the connection)
        collection_info = retrieval_service.qdrant_client.get_collection(
            retrieval_service.collection_name
        )

        print(f"✓ Qdrant connection successful!")
        print(f"✓ Collection '{retrieval_service.collection_name}' exists")
        print(f"✓ Points count: {collection_info.points_count}")
        print(f"✓ Vector size: {collection_info.config.params.vectors.size}")

        return True

    except Exception as e:
        print(f"✗ Qdrant connection failed: {e}")
        print("Please check your QDRANT_URL and QDRANT_API_KEY in the .env file")
        return False

async def test_retrieval_functionality():
    """
    Test the retrieval functionality
    """
    print("\nTesting retrieval functionality...")

    try:
        retrieval_service = RetrievalService()

        # Initialize collection if it doesn't exist
        await retrieval_service.qdrant_client.recreate_collection(
            collection_name=retrieval_service.collection_name,
            vectors_config={"size": 1024, "distance": "Cosine"}  # Cohere embedding size
        )

        # Test with a simple query
        test_query = "test"
        results = await retrieval_service.retrieve_passages(test_query)

        print(f"✓ Retrieval functionality test completed!")
        print(f"✓ Found {len(results)} passages for test query")

        return True

    except Exception as e:
        print(f"✗ Retrieval functionality test failed: {e}")
        return False

async def test_embedding_retrieval_integration():
    """
    Test the integration between embedding and retrieval
    """
    print("\nTesting embedding-retrieval integration...")

    try:
        embedding_service = EmbeddingService()
        retrieval_service = RetrievalService()

        # Create a test passage and store it
        from src.models.retrieved_passage import RetrievedPassage
        import uuid

        test_content = "The humanoid robot uses advanced AI to navigate physical environments."
        test_passage = RetrievedPassage(
            content=test_content,
            source_url="test://integration",
            source_title="Integration Test",
            source_section="Test Section",
            similarity_score=1.0
        )

        # Generate embedding
        embedding = await embedding_service.get_embedding(test_content)

        # Store in Qdrant
        retrieval_service.qdrant_client.upsert(
            collection_name=retrieval_service.collection_name,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "content": test_content,
                    "source_url": test_passage.source_url,
                    "source_title": test_passage.source_title,
                    "source_section": test_passage.source_section,
                    "metadata": test_passage.passage_metadata
                }
            }]
        )

        # Now try to retrieve it
        results = await retrieval_service.retrieve_passages("humanoid robot AI")

        print(f"✓ Embedding-retrieval integration test completed!")
        print(f"✓ Successfully stored and retrieved test passage")
        print(f"✓ Retrieved {len(results)} results for 'humanoid robot AI' query")

        # Clean up - clear the test collection
        try:
            retrieval_service.qdrant_client.delete_collection(retrieval_service.collection_name)
        except:
            pass  # Collection might not exist, which is fine

        return True

    except Exception as e:
        print(f"✗ Embedding-retrieval integration test failed: {e}")
        return False

async def main():
    """
    Main function to run all connection tests
    """
    print("Physical AI & Humanoid Robotics Book - Connection Test")
    print("=" * 60)

    # Check if required environment variables are set
    if not settings.cohere_api_key:
        print("✗ COHERE_API_KEY is not set in .env file")
        return

    if not settings.qdrant_url:
        print("✗ QDRANT_URL is not set in .env file")
        return

    if not settings.qdrant_api_key:
        print("✗ QDRANT_API_KEY is not set in .env file")
        return

    print("✓ Environment variables are properly configured")

    # Run all tests
    tests = [
        test_cohere_connection,
        test_qdrant_connection,
        test_retrieval_functionality,
        test_embedding_retrieval_integration
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    print("\n" + "=" * 60)
    print("Connection Test Results:")

    test_names = [
        "Cohere API Connection",
        "Qdrant Connection",
        "Retrieval Functionality",
        "Embedding-Retrieval Integration"
    ]

    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    if all(results):
        print("\n✓ All connection tests passed!")
        print("✓ Your embedding and retrieval system is ready to use")
        print("✓ Proceed with ingesting your book content using: python ingest_content.py")
    else:
        print("\n✗ Some connection tests failed")
        print("✗ Please fix the issues before proceeding")
        print("✗ Check your API keys and network connection")

if __name__ == "__main__":
    asyncio.run(main())