# RAG Chatbot Setup Instructions

This document provides step-by-step instructions to set up and run the RAG (Retrieval-Augmented Generation) chatbot for the Physical AI & Humanoid Robotics book.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning the repository)

## Step 1: Install Dependencies

First, install the required Python packages:

```bash
cd backend
pip install -r requirements.txt
```

If you don't have the requirements file in the current directory, install the core dependencies:

```bash
pip install fastapi uvicorn cohere qdrant-client psycopg3 python-dotenv pydantic pydantic-settings requests
```

## Step 2: Configure Environment Variables

Create or update the `.env` file in the `backend/` directory with your API keys:

```env
# Cohere API Configuration
COHERE_API_KEY=your_cohere_api_key_here

# Qdrant Cloud Configuration
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=book_content

# Neon Postgres Configuration
NEON_DATABASE_URL=your_neon_database_connection_string_here

# Application Settings
APP_ENV=development
DEBUG=true
```

### Getting Required API Keys:

1. **Cohere API Key**:
   - Go to [Cohere Dashboard](https://dashboard.cohere.ai/)
   - Create an account or log in
   - Navigate to "API Keys" and create a new key
   - Copy the key and paste it in the .env file

2. **Qdrant Cloud**:
   - Go to [Qdrant Cloud](https://qdrant.tech/)
   - Create an account or log in
   - Create a new cluster
   - Copy the URL and API key from your cluster dashboard

3. **Neon Postgres** (Optional for conversation history):
   - Go to [Neon](https://neon.tech/)
   - Create an account or log in
   - Create a new project
   - Copy the connection string from the project dashboard

## Step 3: Ingest Book Content

Before using the chatbot, you need to ingest your book content into the vector database:

```bash
cd backend
python ingest_content.py
```

This script will:
1. Initialize the Qdrant collection
2. Embed your book content using Cohere
3. Store the embeddings in the vector database
4. Create metadata for retrieval

## Step 4: Start the Backend Server

Start the FastAPI server:

```bash
cd backend
python start_server.py
```

Or directly with uvicorn:

```bash
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

## Step 5: Test the API

Once the server is running, you can test it:

1. API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
2. Health Check: [http://localhost:8000/chat/health](http://localhost:8000/chat/health)

You can also test with curl:

```bash
curl -X POST "http://localhost:8000/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is this book about?",
    "mode": "global",
    "selected_text": null,
    "conversation_id": null
  }'
```

## Frontend Integration

The frontend components are already integrated into the Docusaurus site. Make sure your Docusaurus site is configured to connect to the backend server running on `http://localhost:8000`.

## Troubleshooting

### Common Issues:

1. **Port 8000 already in use**:
   - Find the process using port 8000: `lsof -i :8000` (Linux/Mac) or `netstat -ano | findstr :8000` (Windows)
   - Kill the process: `kill -9 <PID>` or `taskkill /PID <PID> /F`

2. **Missing dependencies**:
   - Run: `pip install -r backend/requirements.txt`

3. **Environment variables not set**:
   - Ensure your `.env` file is in the `backend/` directory
   - Verify all required variables are present

4. **API key errors**:
   - Double-check your API keys in the `.env` file
   - Ensure there are no spaces around the `=` sign

5. **Qdrant connection issues**:
   - Verify your Qdrant URL and API key
   - Check that your Qdrant cluster is active
   - Ensure your network allows connections to Qdrant

### Testing the Setup:

Run the test suite to verify everything is working:

```bash
cd backend
python -m pytest tests/ -v
```

## Running Tests

To run the backend tests:

```bash
cd backend
python run_tests.py
```

Or with coverage:

```bash
cd backend
python run_tests.py --coverage
```

## Production Deployment

For production deployment:
1. Set `DEBUG=false` in your `.env` file
2. Use a production-grade database connection
3. Set up proper authentication and rate limiting
4. Use a reverse proxy like Nginx
5. Set up monitoring and logging

## Support

If you encounter issues:
1. Check the logs in your terminal where the server is running
2. Verify all environment variables are correctly set
3. Ensure your API keys have the necessary permissions
4. Confirm your Qdrant and database connections are active

For further assistance, check the project documentation or reach out to the development team.