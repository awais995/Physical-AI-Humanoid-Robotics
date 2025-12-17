#!/usr/bin/env python3
"""
Setup script for the RAG Chatbot for Physical AI & Humanoid Robotics Book
This script will guide you through the complete setup process.
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f" {title} ")
    print("=" * 70)

def check_python_version():
    """Check if Python 3.8+ is installed"""
    print_header("Checking Python Version")

    if sys.version_info < (3, 8):
        print(f"✗ Python 3.8 or higher is required. Current version: {sys.version}")
        return False
    else:
        print(f"✓ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True

def check_and_install_dependencies():
    """Check and install required dependencies"""
    print_header("Installing Dependencies")

    try:
        # Check if backend directory exists
        if not os.path.exists("backend/requirements.txt"):
            print("Creating backend directory and requirements.txt...")
            os.makedirs("backend", exist_ok=True)

            # Create requirements.txt
            requirements = """fastapi==0.104.1
uvicorn==0.24.0
cohere==4.9.3
qdrant-client==1.7.0
psycopg3==3.1.12
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0
requests==2.31.0
asyncio==3.4.3
aiofiles==23.2.1
tiktoken==0.5.2
pytest==7.4.3
pytest-asyncio==0.21.1
"""
            with open("backend/requirements.txt", "w") as f:
                f.write(requirements)

        # Install dependencies
        print("Installing backend dependencies...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Dependencies installed successfully")
            return True
        else:
            print(f"✗ Error installing dependencies: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def check_environment_variables():
    """Check if environment variables are properly set"""
    print_header("Checking Environment Variables")

    required_vars = [
        ("COHERE_API_KEY", "https://dashboard.cohere.ai/"),
        ("QDRANT_URL", "https://cloud.qdrant.io/"),
        ("QDRANT_API_KEY", "https://cloud.qdrant.io/"),
        ("NEON_DATABASE_URL", "https://neon.tech/")
    ]

    missing_vars = []

    # Read the .env file
    env_path = Path("backend/.env")
    if not env_path.exists():
        print("✗ .env file not found in backend directory")
        missing_vars = [var[0] for var in required_vars]
    else:
        with open(env_path, 'r') as f:
            env_content = f.read()

        for var_name, _ in required_vars:
            if f"{var_name}=your_" in env_content or f"{var_name}=" in env_content and var_name + "=" not in env_content.replace(f"{var_name}=", ""):
                # Check if the variable has a placeholder value
                import re
                pattern = f"{var_name}=(.+)"
                matches = re.search(pattern, env_content)
                if matches:
                    value = matches.group(1).strip()
                    if "your_" in value or "placeholder" in value.lower() or value == "":
                        missing_vars.append(var_name)
                else:
                    missing_vars.append(var_name)

    if missing_vars:
        print(f"✗ Missing or placeholder environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these variables in backend/.env file:")
        for var, url in required_vars:
            if var in missing_vars:
                print(f"  {var}=your_actual_key_here  # Get from {url}")
        return False
    else:
        print("✓ All environment variables are properly set")
        return True

def run_connection_tests():
    """Run connection tests"""
    print_header("Running Connection Tests")

    try:
        # Run the connection test script
        result = subprocess.run([
            sys.executable, "backend/test_connection.py"
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"✗ Error running connection tests: {e}")
        return False

def ingest_sample_content():
    """Ingest sample content into the vector database"""
    print_header("Ingesting Sample Content")

    try:
        # Run the ingestion script
        result = subprocess.run([
            sys.executable, "backend/ingest_content.py"
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"✗ Error ingesting content: {e}")
        return False

def start_server():
    """Start the backend server"""
    print_header("Starting Backend Server")

    print("The server will start on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("\nTo start the server manually, run:")
    print("  cd backend")
    print("  python start_server.py")
    print("\nAPI Documentation will be available at: http://localhost:8000/docs")

    try:
        # Start the server
        subprocess.run([
            sys.executable, "backend/start_server.py"
        ])
        return True
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user")
        return True
    except Exception as e:
        print(f"✗ Error starting server: {e}")
        return False

def main():
    """Main setup function"""
    print_header("RAG Chatbot Setup for Physical AI & Humanoid Robotics Book")
    print("This script will guide you through the complete setup process.")

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Install dependencies
    if not check_and_install_dependencies():
        print("\nPlease install the required dependencies before continuing.")
        sys.exit(1)

    # Step 3: Check environment variables
    if not check_environment_variables():
        print("\nPlease configure your environment variables in backend/.env before continuing.")
        print("See backend/SETUP_INSTRUCTIONS.md for detailed instructions.")
        sys.exit(1)

    # Step 4: Run connection tests
    print("\nWould you like to run connection tests? (y/n): ", end="")
    response = input().lower().strip()
    if response in ['y', 'yes']:
        if not run_connection_tests():
            print("\nConnection tests failed. Please fix the issues before continuing.")
            sys.exit(1)

    # Step 5: Ingest sample content
    print("\nWould you like to ingest sample book content? (y/n): ", end="")
    response = input().lower().strip()
    if response in ['y', 'yes']:
        if not ingest_sample_content():
            print("\nContent ingestion failed. Please check your setup.")
            sys.exit(1)

    # Step 6: Start the server
    print("\nWould you like to start the backend server now? (y/n): ", end="")
    response = input().lower().strip()
    if response in ['y', 'yes']:
        start_server()

    print_header("Setup Complete!")
    print("✓ RAG Chatbot is ready to use!")
    print("\nNext steps:")
    print("1. The backend server is running on http://localhost:8000")
    print("2. API documentation: http://localhost:8000/docs")
    print("3. Frontend should now be able to connect to the backend")
    print("4. For full instructions, see: backend/SETUP_INSTRUCTIONS.md")

    print("\nTo restart the server later:")
    print("  cd backend")
    print("  python start_server.py")

if __name__ == "__main__":
    main()