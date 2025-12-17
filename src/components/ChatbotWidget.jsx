import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import ChatInterface from './RagChatbot/ChatInterface';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);

  // Load the chatbot assets only in the browser
  useEffect(() => {
    setHasLoaded(true);
  }, []);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  if (!hasLoaded) {
    return null; // Don't render anything during SSR
  }

  return (
    <BrowserOnly>
      {() => (
        <div style={styles.widgetContainer}>
          {isOpen ? (
            <div style={styles.chatWindow} className="chat-window">
              <div style={styles.chatHeader}>
                <span style={styles.chatTitle}>AI Assistant</span>
                <button
                  onClick={toggleChat}
                  style={styles.closeButton}
                >
                  ×
                </button>
              </div>
              <div style={styles.chatBody}>
                <ChatInterface />
              </div>
            </div>
          ) : null}

          <button
            onClick={toggleChat}
            style={styles.floatingButton}
            className="floating-button"
            aria-label="Open chatbot"
          >
            💬
          </button>
        </div>
      )}
    </BrowserOnly>
  );
};

const styles = {
  widgetContainer: {
    position: 'fixed',
    bottom: '20px',
    right: '20px',
    zIndex: 1000,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  floatingButton: {
    width: '56px',
    height: '56px',
    borderRadius: '50%',
    backgroundColor: 'var(--ifm-color-primary, #3b82f6)',
    color: 'white',
    border: 'none',
    fontSize: '22px',
    cursor: 'pointer',
    boxShadow: '0 4px 16px rgba(59, 130, 246, 0.4)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease',
  },
  floatingButtonHover: {
    transform: 'scale(1.05)',
    backgroundColor: 'var(--ifm-color-primary-dark, #2563eb)',
    boxShadow: '0 6px 20px rgba(59, 130, 246, 0.5)',
  },
  chatWindow: {
    position: 'absolute',
    bottom: '76px',
    right: '0',
    width: '380px',
    maxWidth: '90vw',
    height: '450px',
    maxHeight: '75vh',
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    borderRadius: '16px',
    boxShadow: '0 12px 40px rgba(0, 0, 0, 0.15)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  chatHeader: {
    backgroundColor: 'var(--ifm-color-emphasis-100, #f8fafc)',
    color: 'var(--ifm-font-color-base, #1e293b)',
    padding: '12px 16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderBottom: '1px solid var(--ifm-color-emphasis-200, #e2e8f0)',
  },
  chatTitle: {
    fontWeight: '600',
    fontSize: '15px',
    color: 'var(--ifm-font-color-base, #1e293b)',
  },
  closeButton: {
    background: 'none',
    border: '1px solid var(--ifm-color-emphasis-200, #cbd5e1)',
    borderRadius: '6px',
    color: 'var(--ifm-color-emphasis-600, #64748b)',
    fontSize: '16px',
    cursor: 'pointer',
    padding: '4px',
    width: '32px',
    height: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease',
  },
  closeButtonHover: {
    backgroundColor: 'var(--ifm-color-emphasis-100, #f1f5f9)',
    color: 'var(--ifm-color-emphasis-700, #475569)',
    transform: 'scale(1.05)',
  },
  chatBody: {
    flex: 1,
    overflow: 'hidden',
    backgroundColor: 'var(--ifm-color-emphasis-100, #f8fafc)',
  },
};

// Add CSS animations and responsive styles
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse {
      0% { box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4); }
      50% { box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6); transform: scale(1.05); }
      100% { box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4); }
    }

    @keyframes slideIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 480px) {
      [class*="chatWindow"] {
        width: 95vw !important;
        height: 70vh !important;
        max-height: 400px !important;
        bottom: 80px !important;
      }

      [class*="floatingButton"] {
        width: 56px !important;
        height: 56px !important;
        font-size: 22px !important;
      }
    }

    @media (max-width: 360px) {
      [class*="chatWindow"] {
        width: 98vw !important;
        height: 65vh !important;
        bottom: 75px !important;
      }
    }
  `;
  document.head.appendChild(style);
}

export default ChatbotWidget;