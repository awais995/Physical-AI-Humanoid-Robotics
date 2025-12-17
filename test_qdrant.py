import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('backend/.env')

# Test Qdrant connection
def test_qdrant_connection():
    print("Testing Qdrant connection...")

    # Get settings from environment
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'book_content')

    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection name: {collection_name}")

    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            # For cloud Qdrant, we need to specify https
            https=True if "qdrant.io" in qdrant_url else False
        )

        # Test connection by getting collection info
        try:
            collection_info = client.get_collection(collection_name)
            print(f"SUCCESS: Successfully connected to Qdrant")
            print(f"SUCCESS: Collection '{collection_name}' exists")
            print(f"SUCCESS: Points count: {collection_info.points_count}")
            print(f"SUCCESS: Collection vectors config: {collection_info.config.params}")

            if collection_info.points_count > 0:
                print(f"SUCCESS: Book content appears to be indexed ({collection_info.points_count} points)")

                # Try to retrieve a sample point
                points = client.scroll(
                    collection_name=collection_name,
                    limit=1
                )

                if points[0]:
                    sample_point = points[0][0]
                    print(f"SUCCESS: Sample point ID: {sample_point.id}")
                    print(f"SUCCESS: Sample content preview: {sample_point.payload.get('content', '')[:100]}...")

            else:
                print("ERROR: Collection exists but has no points - book content was not indexed")

        except Exception as e:
            print(f"ERROR: Error accessing collection: {e}")
            # Check if collections exist at all
            try:
                collections = client.get_collections()
                print(f"Available collections: {[col.name for col in collections.collections]}")
            except Exception as e2:
                print(f"ERROR: Error getting collections list: {e2}")

    except Exception as e:
        print(f"ERROR: Error connecting to Qdrant: {e}")
        print("This could be due to:")
        print("1. Invalid QDRANT_URL or QDRANT_API_KEY")
        print("2. Network connectivity issues")
        print("3. Incorrect protocol (http vs https)")


if __name__ == "__main__":
    test_qdrant_connection()