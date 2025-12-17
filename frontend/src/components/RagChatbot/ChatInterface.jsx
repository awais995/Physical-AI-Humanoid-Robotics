import React, { useState, useEffect, useRef } from 'react';
import QueryModeSelector from './QueryModeSelector';
import ResponseRenderer from './ResponseRenderer';
import { chatAPI } from '../../services/chat-api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState('global');
  const [selectedText, setSelectedText] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputText,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Prepare the request based on mode
      const requestData = {
        query_text: inputText,
        mode: selectedMode,
        conversation_id: conversationId
      };

      // Include selected text if in selected-text-only mode
      if (selectedMode === 'selected-text-only' && selectedText) {
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
        citations: response.citations,
        confidence: response.confidence,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request.',
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleTextSelection = () => {
    const selectedText = window.getSelection().toString();
    if (selectedText && selectedText.trim()) {
      setSelectedText(selectedText);
    }
  };

  return (
    <div className="rag-chatbot-interface" style={styles.container}>
      <div style={styles.header}>
        <h3>Book Q&A Assistant</h3>
        <QueryModeSelector
          selectedMode={selectedMode}
          onModeChange={setSelectedMode}
        />
      </div>

      {selectedMode === 'selected-text-only' && selectedText && (
        <div style={styles.selectedTextPreview}>
          <strong>Selected Text:</strong>
          <p>{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}</p>
        </div>
      )}

      <div style={styles.chatContainer}>
        {messages.length === 0 ? (
          <div style={styles.welcomeMessage}>
            <p>Ask me anything about the book! I can answer questions based on the book content.</p>
          </div>
        ) : (
          <div style={styles.messagesContainer}>
            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  ...styles.message,
                  ...(message.role === 'user' ? styles.userMessage : styles.assistantMessage)
                }}
              >
                <div style={styles.messageContent}>
                  {message.role === 'user' ? (
                    <span>{message.content}</span>
                  ) : (
                    <ResponseRenderer
                      content={message.content}
                      citations={message.citations || []}
                      confidence={message.confidence}
                    />
                  )}
                </div>
                <div style={styles.messageTimestamp}>
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            ))}
            {isLoading && (
              <div style={{...styles.message, ...styles.assistantMessage}}>
                <div style={styles.messageContent}>
                  <em>Thinking...</em>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div style={styles.inputContainer}>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the book..."
          style={styles.textArea}
          rows="3"
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          disabled={!inputText.trim() || isLoading}
          style={styles.sendButton}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>

      <div style={styles.instruction}>
        {selectedMode === 'selected-text-only'
          ? 'Select text in the book content, then ask questions about it.'
          : 'Ask any question about the book content.'}
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    border: '1px solid #ddd',
    borderRadius: '8px',
    overflow: 'hidden',
    fontFamily: 'Arial, sans-serif'
  },
  header: {
    padding: '15px',
    backgroundColor: '#f5f5f5',
    borderBottom: '1px solid #ddd',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  chatContainer: {
    flex: 1,
    overflow: 'auto',
    padding: '15px',
    display: 'flex',
    flexDirection: 'column'
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    marginBottom: '15px'
  },
  welcomeMessage: {
    textAlign: 'center',
    padding: '20px',
    color: '#666',
    fontStyle: 'italic'
  },
  message: {
    marginBottom: '15px',
    padding: '10px',
    borderRadius: '8px',
    maxWidth: '80%',
    position: 'relative'
  },
  userMessage: {
    backgroundColor: '#e3f2fd',
    marginLeft: 'auto',
    textAlign: 'right'
  },
  assistantMessage: {
    backgroundColor: '#f5f5f5',
    marginRight: 'auto'
  },
  messageContent: {
    marginBottom: '5px'
  },
  messageTimestamp: {
    fontSize: '0.8em',
    color: '#999',
    textAlign: 'right'
  },
  inputContainer: {
    padding: '15px',
    borderTop: '1px solid #ddd',
    display: 'flex',
    flexDirection: 'column'
  },
  textArea: {
    width: '100%',
    padding: '10px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    resize: 'vertical',
    marginBottom: '10px',
    fontSize: '14px'
  },
  sendButton: {
    alignSelf: 'flex-end',
    padding: '8px 16px',
    backgroundColor: '#1976d2',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px'
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed'
  },
  instruction: {
    padding: '0 15px 15px',
    fontSize: '0.9em',
    color: '#666',
    textAlign: 'center'
  },
  selectedTextPreview: {
    padding: '10px 15px',
    backgroundColor: '#fff3cd',
    border: '1px solid #ffeaa7',
    borderRadius: '4px',
    margin: '0 15px 10px',
    fontSize: '0.9em'
  }
};

export default ChatInterface;