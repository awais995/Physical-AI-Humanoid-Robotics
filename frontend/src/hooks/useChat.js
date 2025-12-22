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
        query_text: query,
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

  // Load conversation history and set messages
  const loadConversationHistory = useCallback(async (convId = null) => {
    const targetConvId = convId || conversationId;
    if (!targetConvId) {
      return null;
    }

    try {
      setIsLoading(true);
      const history = await chatAPI.getConversationHistory(targetConvId);

      // Set the conversation ID and messages
      setConversationId(history.conversation_id);

      // Format messages from history to match our internal format
      const formattedMessages = history.messages.map((msg, index) => ({
        id: index,
        role: msg.role,
        content: msg.content,
        citations: msg.citations || [],
        timestamp: msg.timestamp,
        mode: msg.mode || 'global'
      }));

      setMessages(formattedMessages);
      return history;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [conversationId]);

  // Load list of conversations
  const loadConversations = useCallback(async () => {
    try {
      const convList = await chatAPI.listConversations();
      setConversations(convList);
      return convList;
    } catch (err) {
      setError(err.message);
      return [];
    }
  }, []);

  // Create a new conversation
  const startNewConversation = useCallback(() => {
    setMessages([]);
    setConversationId(null);
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
    loadConversations,
    startNewConversation
  };
};

export default useChat;