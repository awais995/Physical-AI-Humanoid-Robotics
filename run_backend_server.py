#!/usr/bin/env python3
"""
Script to run the RAG Chatbot backend server with proper setup checks
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_setup():
    """Check if the basic setup is complete"""
    print("Checking setup...")

    # Check if backend directory exists
    if not os.path.exists("backend"):
        print("- Backend directory not found!")
        print("Please make sure you're running this from the project root directory.")
        return False

    # Check if requirements are installed
    try:
        import fastapi, uvicorn, cohere, qdrant_client
        print("+ Dependencies are available")
    except ImportError as e:
        print(f"- Missing dependency: {e}")
        print("Please run: pip install -r backend/requirements.txt")
        return False

    # Check if .env file exists and has proper values
    env_path = Path("backend/.env")
    if not env_path.exists():
        print("- .env file not found in backend directory!")
        print("Please create backend/.env with your API keys.")
        return False

    with open(env_path, 'r') as f:
        env_content = f.read()

    # Check for placeholder values
    required_vars = ["COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "NEON_DATABASE_URL"]
    missing_or_placeholder = []

    for var in required_vars:
        if f"{var}=" in env_content:
            import re
            pattern = f"{var}=(.+)"
            matches = re.search(pattern, env_content)
            if matches:
                value = matches.group(1).strip()
                if "your_" in value or "placeholder" in value.lower() or value == "":
                    missing_or_placeholder.append(var)
            else:
                missing_or_placeholder.append(var)
        else:
            missing_or_placeholder.append(var)

    if missing_or_placeholder:
        print(f"- Missing or placeholder values for: {', '.join(missing_or_placeholder)}")
        print("Please update your backend/.env file with actual API keys.")
        return False

    print("+ Environment variables are properly configured")
    return True

def start_server():
    """Start the backend server"""
    print("\nStarting the RAG Chatbot backend server...")
    print("This may take a few moments...\n")

    try:
        # Change to backend directory and start the server
        os.chdir("backend")

        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000"
            # Note: No --reload flag means reload is disabled by default
        ])

        # Wait a bit for the server to start
        time.sleep(3)

        # Check if the process is still running
        if process.poll() is not None:
            print("- Server failed to start. Check the error messages above.")
            return False

        print("+ Server started successfully!")
        print("+ Access the API at: http://localhost:8000")
        print("+ API documentation at: http://localhost:8000/docs")
        print("\nTo stop the server, press Ctrl+C\n")

        try:
            # Wait for the process to finish (this will block until Ctrl+C)
            process.wait()
        except KeyboardInterrupt:
            print("\n\nStopping server...")
            process.terminate()
            try:
                process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if it doesn't shut down gracefully
            print("Server stopped.")
            return True

    except FileNotFoundError:
        print("- uvicorn not found. Please install it:")
        print("  pip install uvicorn")
        return False
    except Exception as e:
        print(f"- Error starting server: {e}")
        return False
    finally:
        # Change back to original directory
        os.chdir("..")

def test_server_health():
    """Test if the server is responding"""
    try:
        response = requests.get("http://localhost:8000/chat/health", timeout=10)
        if response.status_code == 200:
            print("+ Server is responding correctly")
            return True
        else:
            print(f"- Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("- Server is not responding. It may still be starting up.")
        return False
    except Exception as e:
        print(f"- Error testing server: {e}")
        return False

def main():
    """Main function"""
    print("Physical AI & Humanoid Robotics Book - RAG Chatbot Server")
    print("=" * 65)

    # Check setup
    if not check_setup():
        print("\nPlease complete the setup before starting the server.")
        print("You can use: python setup_rag_chatbot.py to guide you through the setup.")
        sys.exit(1)

    # Start the server
    success = start_server()

    if success:
        print("\n" + "=" * 65)
        print("Server has been stopped.")
        print("To start again, run: python run_backend_server.py")
    else:
        print("\n" + "=" * 65)
        print("Server startup failed.")
        print("Check the error messages and try again.")

if __name__ == "__main__":
    main()