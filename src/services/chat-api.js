// Chat API service for interacting with the backend RAG chatbot
// Using global window object to access API base URL
// Check for environment-specific configuration first
const API_BASE_URL = (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_BASE_URL) ||
                     (typeof window !== 'undefined' && window.chatbotConfig && window.chatbotConfig.apiBaseUrl) ||
                     'http://localhost:8000';

// Function to check if we're in a browser environment
const isBrowser = typeof window !== 'undefined';

// For production deployments, provide a fallback or configurable URL
// If in development and using Docusaurus, the API_BASE_URL can be configured via environment or the config file

class ChatAPI {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async queryChat(requestData) {
    try {
      const response = await fetch(`${this.baseURL}/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error querying chat:', error);
      throw error;
    }
  }

  async getConversationHistory(conversationId) {
    try {
      const response = await fetch(`${this.baseURL}/chat/conversation/${conversationId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error getting conversation history:', error);
      throw error;
    }
  }

  async listConversations() {
    try {
      const response = await fetch(`${this.baseURL}/chat/conversations`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error listing conversations:', error);
      throw error;
    }
  }
}

export const chatAPI = new ChatAPI();
export default chatAPI;