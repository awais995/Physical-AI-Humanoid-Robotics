import React, { useState, useEffect } from 'react';
import { useChat } from '../../hooks/useChat';

const ConversationHistory = ({ onSelectConversation, currentConversationId }) => {
  const { conversations, loadConversations, loadConversationHistory, startNewConversation } = useChat();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchConversations = async () => {
      setIsLoading(true);
      try {
        await loadConversations();
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchConversations();
  }, [loadConversations]);

  const handleSelectConversation = async (conversationId) => {
    try {
      await loadConversationHistory(conversationId);
      if (onSelectConversation) {
        onSelectConversation(conversationId);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const handleNewConversation = () => {
    startNewConversation();
    if (onSelectConversation) {
      onSelectConversation(null);
    }
  };

  return (
    <div className="conversation-history">
      <div className="conversation-header">
        <h3>Chat History</h3>
        <button
          className="new-conversation-btn"
          onClick={handleNewConversation}
          title="Start new conversation"
        >
          + New Chat
        </button>
      </div>

      {error && (
        <div className="error-message">
          Error loading conversations: {error}
        </div>
      )}

      {isLoading ? (
        <div className="loading">Loading conversations...</div>
      ) : (
        <div className="conversation-list">
          {conversations && conversations.length > 0 ? (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={`conversation-item ${
                  currentConversationId === conv.id ? 'active' : ''
                }`}
                onClick={() => handleSelectConversation(conv.id)}
              >
                <div className="conversation-title">
                  {conv.title || `Conversation ${conv.id.substring(0, 8)}...`}
                </div>
                <div className="conversation-meta">
                  <small>
                    {new Date(conv.created_at).toLocaleDateString()} •{' '}
                    {conv.last_message?.substring(0, 50)}...
                  </small>
                </div>
              </div>
            ))
          ) : (
            <div className="no-conversations">
              No conversation history yet. Start a new chat!
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ConversationHistory;