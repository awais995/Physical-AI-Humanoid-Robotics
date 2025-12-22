#!/usr/bin/env python3
"""
Test script to check Qdrant client methods with actual configuration
"""
import sys
import os

# Add the backend src directory to the path
backend_src = os.path.join(os.path.dirname(__file__), 'backend', 'src')
sys.path.insert(0, backend_src)

from config.settings import settings
from qdrant_client import QdrantClient

def test_qdrant_client():
    print("Creating Qdrant client with production configuration...")

    # Create client as it's done in the code
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        https=True if "qdrant.io" in settings.qdrant_url else False
    )

    print(f"Qdrant client type: {type(qdrant_client)}")
    print(f"Has search method: {hasattr(qdrant_client, 'search')}")
    print(f"Has retrieve method: {hasattr(qdrant_client, 'retrieve')}")

    # Test with a simple method that should always exist
    print(f"Has get_collections method: {hasattr(qdrant_client, 'get_collections')}")

    # List all methods to see what's available
    all_methods = [attr for attr in dir(qdrant_client) if not attr.startswith('_') and callable(getattr(qdrant_client, attr))]
    print(f"\nAll callable methods (first 20): {all_methods[:20]}")

    # Specifically look for search-related methods
    search_related = [m for m in all_methods if 'search' in m.lower()]
    retrieve_related = [m for m in all_methods if 'retrieve' in m.lower()]
    print(f"\nSearch-related methods: {search_related}")
    print(f"Retrieve-related methods: {retrieve_related}")

if __name__ == "__main__":
    test_qdrant_client()