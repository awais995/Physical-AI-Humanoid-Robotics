import React, { useState, useEffect, useRef } from 'react';
import ResponseRenderer from './ResponseRenderer';
import { chatAPI } from '../../services/chat-api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [selectedMode, setSelectedMode] = useState('global');
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Set up text selection listener
  useEffect(() => {
    const handleTextSelection = () => {
      const selection = window.getSelection();
      const selectedTextContent = selection?.toString()?.trim();

      // Only update if there's actually selected text
      if (selectedTextContent) {
        setSelectedText(selectedTextContent);
        // Automatically switch to selected-text-only mode when text is selected
        if (selectedMode !== 'selected-text-only') {
          setSelectedMode('selected-text-only');
        }
      } else {
        // Clear selection when no text is selected
        setSelectedText('');
      }
    };

    // Use more specific events for better performance
    document.addEventListener('mouseup', handleTextSelection);
    document.addEventListener('touchend', handleTextSelection);
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Escape') {
        // Clear selection when Escape is pressed
        setSelectedText('');
        // Switch back to global mode if it was in selected-text-only mode
        if (selectedMode === 'selected-text-only') {
          setSelectedMode('global');
        }
      } else {
        handleTextSelection();
      }
    });

    return () => {
      document.removeEventListener('mouseup', handleTextSelection);
      document.removeEventListener('touchend', handleTextSelection);
      document.removeEventListener('keyup', handleTextSelection);
    };
  }, [selectedMode]); // Add selectedMode as dependency

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
      // Prepare the request with selected mode and selected text if in selected-text-only mode
      const requestData = {
        query_text: inputText,
        mode: selectedMode,
        conversation_id: conversationId
      };

      // Include selected text only if in selected-text-only mode
      if (selectedMode === 'selected-text-only' && selectedText) {
        requestData.selected_text = selectedText;
      } else if (selectedMode === 'selected-text-only' && !selectedText) {
        // If user is in selected-text-only mode but no text is selected, show an error
        const errorMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: 'Please select some text on the page first, then ask your question about it.',
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, errorMessage]);
        setIsLoading(false);
        return;
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

      // Create a more informative error message based on the type of error
      let errorMessageContent = 'Sorry, I encountered an error while processing your request.';

      // Check if it's a network error (most likely API connection issue)
      if (error.message.includes('fetch') || error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorMessageContent = 'Unable to connect to the AI service. Please check if the backend server is running and properly configured.';
      } else if (error.message.includes('404') || error.message.includes('405')) {
        errorMessageContent = 'The AI service endpoint is not available. Please check the API configuration.';
      } else if (error.message.includes('500')) {
        errorMessageContent = 'The AI service encountered an internal error. Please try again later.';
      } else if (error.message.includes('ETIMEDOUT') || error.message.includes('timeout')) {
        errorMessageContent = 'The request timed out. Please check your connection and try again.';
      }

      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: errorMessageContent,
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

  return (
    <div style={styles.container}>

      {/* Chat Messages */}
      <div style={styles.chatArea}>
        {messages.length === 0 ? (
          <div style={styles.welcome}>
            <div style={styles.welcomeIcon}>🤖</div>
            <h3 style={styles.welcomeTitle}>Welcome to AI Assistant</h3>
            <p style={styles.welcomeText}>Hello! Select text on the page to ask questions about it, or ask general questions in global mode.</p>
          </div>
        ) : (
          <div style={styles.messages}>
            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  ...styles.message,
                  ...(message.role === 'user' ? styles.userMessage : styles.assistantMessage)
                }}
              >
                {message.role === 'assistant' && (
                  <div style={styles.avatarSmall}>🤖</div>
                )}
                <div style={styles.messageContent}>
                  <div style={{
                    ...styles.messageText,
                    ...(message.role === 'user'
                      ? styles.userMessageText
                      : styles.assistantMessageText)
                  }}>
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
                  <div style={styles.messageTime}>
                    {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
                {message.role === 'user' && (
                  <div style={styles.avatarSmall}>👤</div>
                )}
              </div>
            ))}
            {isLoading && (
              <div style={styles.message}>
                <div style={styles.avatarSmall}>🤖</div>
                <div style={styles.messageContent}>
                  <div style={styles.typingIndicator}>
                    <span style={styles.dot}></span>
                    <span style={styles.dot}></span>
                    <span style={styles.dot}></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Mode Selector and Input Area */}
      <div style={styles.inputContainer}>
        {/* Query Mode Selector - Horizontal and Compact */}
        <div style={styles.modeSelectorContainer}>
          <button
            style={{
              ...styles.modeButton,
              ...(selectedMode === 'global' ? styles.modeButtonActive : styles.modeButtonInactive)
            }}
            onClick={() => {
              setSelectedMode('global');
              // Clear selected text when switching to global mode
              setSelectedText('');
            }}
          >
            Global
          </button>
          <button
            style={{
              ...styles.modeButton,
              ...(selectedMode === 'selected-text-only' ? styles.modeButtonActive : styles.modeButtonInactive)
            }}
            onClick={() => setSelectedMode('selected-text-only')}
          >
            Selected Text
          </button>
        </div>

        {/* Selected Text Indicator */}
        {selectedMode === 'selected-text-only' && selectedText && (
          <div style={styles.selectedTextIndicator}>
            <span style={styles.selectedTextCount}>
              Selected: {selectedText ? selectedText.substring(0, 50) : ''}{selectedText && selectedText.length > 50 ? '...' : ''}
            </span>
          </div>
        )}

        {/* Input Area */}
        <div style={styles.inputArea}>
          <button style={styles.inputActionButton}>+</button>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedMode === 'selected-text-only' && !selectedText
                ? "Select text on the page first..."
                : "Type your message..."
            }
            style={styles.textarea}
            rows="1"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || isLoading}
            style={{
              ...styles.sendButton,
              ...(isLoading ? styles.sendButtonDisabled : {})
            }}
          >
            <span style={styles.sendIcon}>➤</span>
          </button>
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '400px',
    width: '100%',
    margin: '0 auto',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    color: 'var(--ifm-font-color-base, #1e293b)',
    borderRadius: '12px',
    overflow: 'hidden',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    border: '1px solid var(--ifm-color-emphasis-200, #e2e8f0)',
    position: 'relative',
    minHeight: '400px'
  },
  inputContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    padding: '12px',
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    borderTop: '1px solid var(--ifm-color-emphasis-200, #e2e8f0)',
    flexShrink: 0
  },
  modeSelectorContainer: {
    display: 'flex',
    gap: '6px',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '4px',
    backgroundColor: 'var(--ifm-color-emphasis-100, #f1f5f9)',
    borderRadius: '10px',
    border: '1px solid var(--ifm-color-emphasis-200, #cbd5e1)'
  },
  modeButton: {
    padding: '5px 10px',
    fontSize: '12px',
    fontWeight: '500',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    minWidth: '70px'
  },
  modeButtonActive: {
    backgroundColor: 'var(--ifm-color-primary, #3b82f6)',
    color: 'white'
  },
  modeButtonInactive: {
    backgroundColor: 'transparent',
    color: 'var(--ifm-font-color-base, #1e293b)',
    opacity: '0.7'
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 20px',
    backgroundColor: '#f8fafc',
    borderBottom: '1px solid #e2e8f0',
    flexShrink: 0
  },
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px'
  },
  avatar: {
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    backgroundColor: '#3b82f6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '18px',
    fontWeight: 'bold',
    boxShadow: '0 2px 6px rgba(59, 130, 246, 0.3)'
  },
  headerText: {
    display: 'flex',
    flexDirection: 'column'
  },
  title: {
    margin: 0,
    fontSize: '16px',
    fontWeight: '600',
    color: '#1e293b'
  },
  status: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '12px',
    color: '#64748b'
  },
  statusIndicator: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: '#10b981'
  },
  statusText: {
    fontSize: '12px',
    color: '#64748b'
  },
  headerActions: {
    display: 'flex',
    gap: '8px'
  },
  actionButton: {
    width: '36px',
    height: '36px',
    borderRadius: '8px',
    border: '1px solid #e2e8f0',
    backgroundColor: '#f1f5f9',
    color: '#475569',
    fontSize: '14px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease'
  },
  actionButtonHover: {
    backgroundColor: '#e2e8f0',
    transform: 'scale(1.05)'
  },
  chatArea: {
    flex: 1,
    overflowY: 'auto',
    padding: '12px',
    backgroundColor: 'var(--ifm-color-emphasis-100, #f8fafc)',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '0'
  },
  welcome: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    padding: '20px',
    color: 'var(--ifm-color-emphasis-600, #64748b)',
    backgroundColor: 'transparent'
  },
  welcomeIcon: {
    fontSize: '48px',
    marginBottom: '12px',
    opacity: 0.8
  },
  welcomeTitle: {
    margin: '0 0 8px 0',
    fontSize: '18px',
    fontWeight: '600',
    color: 'var(--ifm-font-color-base, #1e293b)',
    marginBottom: '6px'
  },
  welcomeText: {
    margin: 0,
    fontSize: '13px',
    color: 'var(--ifm-color-emphasis-600, #64748b)',
    lineHeight: '1.4',
    maxWidth: '100%',
    padding: '0 8px'
  },
  messages: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    flex: 1
  },
  message: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '10px',
    maxWidth: '90%',
    animation: 'fadeIn 0.3s ease-out',
    flexShrink: 0
  },
  userMessage: {
    alignSelf: 'flex-end',
    flexDirection: 'row-reverse'
  },
  messageContent: {
    display: 'flex',
    flexDirection: 'column',
    minWidth: '80px',
    flex: 1
  },
  messageText: {
    padding: '10px 14px',
    borderRadius: '16px',
    fontSize: '13px',
    lineHeight: '1.4',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
    maxWidth: '100%'
  },
  userMessageText: {
    backgroundColor: 'var(--ifm-color-primary, #3b82f6)',
    color: 'white',
    borderBottomRightRadius: '4px'
  },
  assistantMessageText: {
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    color: 'var(--ifm-font-color-base, #1e293b)',
    border: '1px solid var(--ifm-color-emphasis-200, #e2e8f0)',
    borderBottomLeftRadius: '4px'
  },
  messageTime: {
    fontSize: '10px',
    color: 'var(--ifm-color-emphasis-500, #94a3b8)',
    marginTop: '3px',
    textAlign: 'right'
  },
  avatarSmall: {
    width: '28px',
    height: '28px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '12px',
    flexShrink: 0,
    fontWeight: '500'
  },
  typingIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '3px',
    padding: '8px 12px',
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    border: '1px solid var(--ifm-color-emphasis-200, #e2e8f0)',
    borderRadius: '16px',
    borderBottomLeftRadius: '4px',
    alignSelf: 'flex-start'
  },
  dot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    backgroundColor: 'var(--ifm-color-emphasis-500, #94a3b8)',
    animation: 'bounce 1.4s infinite'
  },
  inputArea: {
    display: 'flex',
    alignItems: 'center',
    padding: '8px 0',
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    flexShrink: 0,
    flexWrap: 'nowrap'
  },
  inputActionButton: {
    width: '36px',
    height: '36px',
    borderRadius: '8px',
    border: '1px solid var(--ifm-color-emphasis-200, #cbd5e1)',
    backgroundColor: 'var(--ifm-color-emphasis-100, #f1f5f9)',
    color: 'var(--ifm-color-emphasis-600, #475569)',
    fontSize: '14px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: '8px',
    transition: 'all 0.2s ease',
    flexShrink: 0
  },
  inputActionButtonHover: {
    backgroundColor: 'var(--ifm-color-emphasis-200, #e2e8f0)',
    transform: 'scale(1.05)'
  },
  textarea: {
    flex: 1,
    padding: '10px 14px',
    border: '1px solid var(--ifm-color-emphasis-200, #cbd5e1)',
    borderRadius: '18px',
    resize: 'none',
    outline: 'none',
    fontSize: '13px',
    minHeight: '40px',
    maxHeight: '100px',
    overflowY: 'auto',
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    color: 'var(--ifm-font-color-base, #1e293b)',
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05) inset',
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
    minWidth: '0'
  },
  textareaFocus: {
    border: '1px solid var(--ifm-color-primary, #3b82f6)',
    boxShadow: '0 0 0 2px rgba(59, 130, 246, 0.1)'
  },
  sendButton: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    border: 'none',
    backgroundColor: 'var(--ifm-color-primary, #3b82f6)',
    color: 'white',
    fontSize: '14px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: '8px',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 4px rgba(59, 130, 246, 0.3)',
    flexShrink: 0
  },
  sendButtonHover: {
    backgroundColor: 'var(--ifm-color-primary-dark, #2563eb)',
    transform: 'scale(1.05)',
    boxShadow: '0 3px 6px rgba(59, 130, 246, 0.4)'
  },
  sendButtonDisabled: {
    backgroundColor: 'var(--ifm-color-emphasis-200, #cbd5e1)',
    cursor: 'not-allowed',
    boxShadow: 'none'
  },
  sendIcon: {
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  selectedTextIndicator: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '4px 0',
    width: '100%'
  },
  selectedTextCount: {
    fontSize: '11px',
    color: 'var(--ifm-color-emphasis-600, #64748b)',
    backgroundColor: 'var(--ifm-color-emphasis-100, #f1f5f9)',
    padding: '4px 8px',
    borderRadius: '12px',
    maxWidth: '100%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
  }
};

// Add CSS animations and responsive styles
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes bounce {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-5px); }
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 480px) {
      [class*="container"] {
        height: 400px !important;
        max-width: 95vw !important;
      }

      [class*="welcomeText"] {
        font-size: 14px !important;
        padding: 0 8px !important;
      }

      [class*="messageText"] {
        padding: 12px 16px !important;
        font-size: 14px !important;
      }

      [class*="inputArea"] {
        padding: 12px 16px !important;
      }

      [class*="textarea"] {
        padding: 12px 16px !important;
        font-size: 14px !important;
      }

      [class*="inputActionButton"],
      [class*="sendButton"] {
        width: 40px !important;
        height: 40px !important;
        font-size: 16px !important;
      }
    }

    @media (max-width: 360px) {
      [class*="container"] {
        height: 360px !important;
      }

      [class*="welcomeTitle"] {
        font-size: 20px !important;
      }

      [class*="welcomeText"] {
        font-size: 13px !important;
      }

      [class*="messageText"] {
        font-size: 13px !important;
        padding: 10px 14px !important;
      }
    }
  `;
  document.head.appendChild(style);
}

export default ChatInterface;