import { useState, useEffect, useCallback } from 'react';
import { chatAPI } from '../services/chat-api';

const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [conversations, setConversations] = useState([]); // Store list of conversations

  // Initialize with any existing conversation
  const initializeConversation = useCallback((initialConversationId = null) => {
    if (initialConversationId) {
      setConversationId(initialConversationId);
    }
  }, []);

  // Send a message to the chatbot
  const sendMessage = useCallback(async (query, mode = 'global', selectedText = null) => {
    if (!query.trim() || isLoading) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Add user message to chat
      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: query,
        timestamp: new Date().toISOString(),
        mode: mode
      };

      setMessages(prev => [...prev, userMessage]);

      // Prepare the request based on mode
      const requestData = {
        query: query,
        mode: mode,
        conversation_id: conversationId
      };

      // Include selected text if in selected-text-only mode
      if (mode === 'selected-text-only' && selectedText) {
        requestData.selected_text = selectedText;
      }

      // Send query to backend
      const response = await chatAPI.queryChat(requestData);

      // Update conversation ID if new conversation was created
      if (response.conversation_id && !conversationId) {
        setConversationId(response.conversation_id);
      }

      // Add AI response to chat
      const aiMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.response,
        citations: response.citations || [],
        confidence: response.confidence,
        mode: response.mode,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      setError(err.message);

      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${err.message}`,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [conversationId, isLoading]);

  // Clear the current conversation
  const clearConversation = useCallback(() => {
    setMessages([]);
    setConversationId(null);
    setError(null);
  }, []);

  // Get conversation history (if conversation ID exists)
  const loadConversationHistory = useCallback(async () => {
    if (!conversationId) {
      return [];
    }

    try {
      const history = await chatAPI.getConversationHistory(conversationId);
      setMessages(history.messages || []);
      return history;
    } catch (err) {
      setError(err.message);
      return [];
    }
  }, [conversationId]);

  // Load all conversations for the user
  const loadAllConversations = useCallback(async () => {
    try {
      const convList = await chatAPI.listConversations();
      setConversations(convList);
      return convList;
    } catch (err) {
      setError(err.message);
      return [];
    }
  }, []);

  // Switch to a different conversation
  const switchConversation = useCallback(async (convId) => {
    setConversationId(convId);
    setMessages([]); // Clear current messages
    await loadConversationHistory(convId); // Load the selected conversation's history
  }, [loadConversationHistory]);

  // Create a new conversation
  const createNewConversation = useCallback(() => {
    setConversationId(null);
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    conversationId,
    conversations,
    initializeConversation,
    sendMessage,
    clearConversation,
    loadConversationHistory,
    loadAllConversations,
    switchConversation,
    createNewConversation
  };
};

export default useChat;