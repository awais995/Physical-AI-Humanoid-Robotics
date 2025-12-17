import requests
import json
import time

# Test the backend API with more detailed error checking
def test_backend_detailed():
    base_url = "http://localhost:8000"

    # Test health endpoint first
    try:
        health_response = requests.get(f"{base_url}/health")
        print(f"SUCCESS: Health check: {health_response.status_code} - {health_response.json()}")
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        return

    # Test if Qdrant is accessible by trying a simple query
    print("\nTesting retrieval process...")
    try:
        test_payload = {
            "query": "What is this book about?",
            "mode": "global",
            "selected_text": None,
            "conversation_id": None
        }

        headers = {
            "Content-Type": "application/json"
        }

        print("Sending query to backend...")
        chat_response = requests.post(f"{base_url}/chat/query",
                                     data=json.dumps(test_payload),
                                     headers=headers)

        print(f"Response status: {chat_response.status_code}")
        if chat_response.status_code == 200:
            response_data = chat_response.json()
            print(f"SUCCESS: Response: {response_data['response'][:200]}...")
            print(f"SUCCESS: Citations: {len(response_data.get('citations', []))} citations")
            print(f"SUCCESS: Confidence: {response_data.get('confidence', 'N/A')}")
            print(f"SUCCESS: Conversation ID: {response_data.get('conversation_id')}")
        else:
            print(f"ERROR: Error response: {chat_response.text}")

            # Let's also test a simple retrieval without generation to isolate the issue
            print("\nTrying to test just the retrieval process...")
            try:
                # We'll need to add a test endpoint for this, but for now let's see if it's a
                # Cohere API issue by checking if the embeddings are working
                print("The error might be related to:")
                print("1. Invalid Cohere API key")
                print("2. Invalid Qdrant connection")
                print("3. Network connectivity issues")
                print("4. The book content wasn't properly indexed")
            except:
                pass

    except requests.exceptions.ConnectionError:
        print("ERROR: Connection error: Cannot connect to the backend server")
    except Exception as e:
        print(f"ERROR: Chat query failed with exception: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting completed.")

if __name__ == "__main__":
    test_backend_detailed()