import subprocess
import sys
import os
import signal
import time
from threading import Thread
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv('backend/.env')

def check_dependencies():
    """
    Check if required dependencies are installed
    """
    try:
        import fastapi
        import uvicorn
        import cohere
        import qdrant_client
        import python_dotenv
        import pydantic
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False

def check_environment_variables():
    """
    Check if required environment variables are set
    """
    required_vars = ['COHERE_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY', 'NEON_DATABASE_URL']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"✗ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False

    print("✓ All required environment variables are set")
    return True

def is_port_available(port):
    """
    Check if a port is available
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def start_backend_server():
    """
    Start the backend server using uvicorn
    """
    print("Starting backend server...")

    # Check dependencies and environment
    if not check_dependencies():
        return False

    if not check_environment_variables():
        return False

    # Check if port 8000 is available
    if not is_port_available(8000):
        print("✗ Port 8000 is already in use. Please stop any existing server first.")
        return False

    # Start the server
    try:
        # Change to the backend directory
        os.chdir('backend')

        # Start the server using uvicorn
        import uvicorn
        from src.main import app

        print("✓ Server is starting on http://localhost:8000")
        print("✓ Waiting for server to be ready...")

        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

    except Exception as e:
        print(f"✗ Error starting server: {e}")
        return False

def start_server_with_subprocess():
    """
    Start the server using subprocess (alternative method)
    """
    try:
        # Change to the backend directory
        original_dir = os.getcwd()
        os.chdir('backend')

        # Run uvicorn command
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])

        print("✓ Server started successfully!")
        print("✓ Access the API at: http://localhost:8000")
        print("✓ API documentation at: http://localhost:8000/docs")

        # Wait for the process to finish
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n✓ Shutting down server...")
            process.terminate()
            process.wait()

    except Exception as e:
        print(f"✗ Error starting server: {e}")
        return False
    finally:
        os.chdir(original_dir)

def test_server_health():
    """
    Test if the server is responding
    """
    try:
        response = requests.get("http://localhost:8000/chat/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is responding correctly")
            return True
        else:
            print(f"✗ Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Server is not responding. Make sure it's running on port 8000.")
        return False
    except Exception as e:
        print(f"✗ Error testing server: {e}")
        return False

def main():
    """
    Main function to start the server
    """
    print("Physical AI & Humanoid Robotics Book - Backend Server")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("\nPlease install required dependencies first:")
        print("pip install -r backend/requirements.txt")
        return

    # Check environment variables
    if not check_environment_variables():
        print("\nPlease configure your .env file with the required API keys:")
        print("COHERE_API_KEY=your_cohere_api_key")
        print("QDRANT_URL=your_qdrant_cloud_url")
        print("QDRANT_API_KEY=your_qdrant_api_key")
        print("NEON_DATABASE_URL=your_neon_database_url")
        return

    print("\nStarting server...")
    print("Press Ctrl+C to stop the server\n")

    # Start the server
    start_server_with_subprocess()

if __name__ == "__main__":
    main()