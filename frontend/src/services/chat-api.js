// Chat API service for interacting with the backend RAG chatbot
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'https://awaissoomro-chat-bot.hf.space';

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
      const response = await fetch(`${this.baseURL}/chat/conversations/${conversationId}`, {
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