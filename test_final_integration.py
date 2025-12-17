#!/usr/bin/env python3
"""
Final integration test for Qdrant vector store with the RAG chatbot
"""
import requests
import json
import time

def test_backend_health():
    """Test if the backend server is healthy"""
    print("Testing backend health...")
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✓ Backend is healthy")
        return True
    else:
        print(f"✗ Backend health check failed: {response.status_code}")
        return False

def test_chat_health():
    """Test if the chat endpoint is healthy"""
    print("Testing chat health...")
    response = requests.get("http://localhost:8000/chat/health")
    if response.status_code == 200:
        print("✓ Chat endpoint is healthy")
        return True
    else:
        print(f"✗ Chat health check failed: {response.status_code}")
        return False

def test_qdrant_integration():
    """Test Qdrant integration with a meaningful query"""
    print("Testing Qdrant integration...")

    # Test query about ROS 2
    response = requests.post(
        "http://localhost:8000/chat/query",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "query_text": "What is ROS 2?",
            "mode": "global",
            "conversation_id": "integration_test"
        })
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Query successful")
        print(f"  - Response length: {len(data['response'])} characters")
        print(f"  - Retrieved passages: {data['retrieved_passages_count']}")
        print(f"  - Confidence score: {data['confidence']:.3f}")

        # Check if we got meaningful results
        if data['retrieved_passages_count'] > 0 and data['confidence'] > 0.5:
            print("✓ Qdrant integration is working properly")
            return True
        else:
            print("✗ Qdrant integration returned poor results")
            return False
    else:
        print(f"✗ Query failed: {response.status_code} - {response.text}")
        return False

def test_context_awareness():
    """Test that the system correctly handles queries outside the book context"""
    print("Testing context awareness...")

    # Test query about Qdrant (not in the book)
    response = requests.post(
        "http://localhost:8000/chat/query",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "query_text": "Explain Qdrant vector database",
            "mode": "global",
            "conversation_id": "context_test"
        })
    )

    if response.status_code == 200:
        data = response.json()
        # Check if the response acknowledges the lack of context
        response_text = data['response'].lower()
        if "not mentioned" in response_text or "cannot provide" in response_text or "based on the given context" in response_text:
            print("✓ System correctly handles out-of-context queries")
            print(f"  - Low confidence score: {data['confidence']:.3f}")
            return True
        else:
            print("✗ System should have recognized this is outside its context")
            return False
    else:
        print(f"✗ Context test failed: {response.status_code}")
        return False

def main():
    """Run all integration tests"""
    print("Running final integration tests for Qdrant vector store...")
    print("=" * 60)

    tests = [
        ("Backend Health", test_backend_health),
        ("Chat Health", test_chat_health),
        ("Qdrant Integration", test_qdrant_integration),
        ("Context Awareness", test_context_awareness),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            if test_func():
                passed += 1
                print(f"  Result: PASSED")
            else:
                print(f"  Result: FAILED")
        except Exception as e:
            print(f"  Result: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"Integration tests: {passed}/{total} passed")

    if passed == total:
        print("🎉 All integration tests PASSED! Qdrant integration is working perfectly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    main()