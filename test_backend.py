import requests
import json

# Test the backend API
def test_backend():
    base_url = "https://awaissoomro-chat-bot.hf.space"

    # Test health endpoint first
    try:
        health_response = requests.get(f"{base_url}/health")
        print(f"Health check: {health_response.status_code} - {health_response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Test chat endpoint
    try:
        test_payload = {
            "query_text": "What is this book about?",
            "mode": "global",
            "selected_text": None,
            "conversation_id": None
        }

        headers = {
            "Content-Type": "application/json"
        }

        chat_response = requests.post(f"{base_url}/chat/query",
                                     data=json.dumps(test_payload),
                                     headers=headers)

        print(f"Chat query response: {chat_response.status_code}")
        if chat_response.status_code == 200:
            response_data = chat_response.json()
            print(f"Response: {response_data['response'][:200]}...")
            print(f"Citations: {len(response_data.get('citations', []))} citations")
            print(f"Confidence: {response_data.get('confidence', 'N/A')}")
        else:
            print(f"Error: {chat_response.text}")

    except Exception as e:
        print(f"Chat query failed: {e}")

if __name__ == "__main__":
    test_backend()