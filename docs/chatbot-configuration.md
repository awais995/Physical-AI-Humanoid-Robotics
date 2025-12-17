# Chatbot Configuration Guide

This guide explains how to properly configure the chatbot API for different deployment scenarios.

## Configuration Options

The chatbot can be configured in multiple ways depending on your deployment environment:

### 1. Environment Variables (Recommended for Build-Time Configuration)

Set the API URL during the build process:

```bash
# For React/Docusaurus builds
REACT_APP_API_BASE_URL=https://your-backend-domain.com npm run build
```

### 2. Runtime Configuration via Window Object

You can configure the API URL by setting the `chatbotConfig` object before the chatbot script loads:

```html
<script>
  // Set this BEFORE the chatbot-config.js script loads
  window.chatbotConfig = {
    apiBaseUrl: 'https://your-backend-domain.com'
  };
</script>
<script src="/js/chatbot-config.js"></script>
```

### 3. Default Configuration

By default, the chatbot will try to connect to `http://localhost:8000`, which is suitable for local development.

## Deployment Scenarios

### Local Development
- The default `http://localhost:8000` works if you have a backend server running locally on port 8000
- Make sure your backend API server is running before testing the chatbot

### Production Deployment
- Update the `chatbot-config.js` file or use environment variables to point to your production backend
- Ensure your backend API is accessible from the deployed frontend

### Vercel/Netlify Deployment
- Use environment variables during the build process
- Example for Vercel: Set `REACT_APP_API_BASE_URL` in your project settings

## Troubleshooting Common Issues

### "Unable to connect to the AI service"
- Check if your backend server is running
- Verify the API URL is correct and accessible
- Ensure CORS is properly configured on your backend

### "The AI service endpoint is not available"
- Verify that the `/chat/query` endpoint exists on your backend
- Check that your backend server is properly configured to handle requests

## API Endpoint Requirements

Your backend must provide the following endpoint:
- POST `/chat/query` - Handles chat queries and returns responses with optional citations

Example request body:
```json
{
  "query": "Your question here",
  "mode": "global", // or "selected-text-only"
  "conversation_id": "optional-conversation-id",
  "selected_text": "optional-selected-text"
}
```

Example response:
```json
{
  "response": "AI response here",
  "citations": ["citation1", "citation2"],
  "confidence": 0.95,
  "conversation_id": "new-or-existing-conversation-id"
}
```