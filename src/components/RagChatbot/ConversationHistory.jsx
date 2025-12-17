import React from 'react';

const ConversationHistory = ({ conversations, onConversationSelect, onCreateNew }) => {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3>Chat History</h3>
        <button onClick={onCreateNew} style={styles.newButton}>
          + New Chat
        </button>
      </div>

      {conversations && conversations.length > 0 ? (
        <div style={styles.historyList}>
          {conversations.map((conv) => (
            <div
              key={conv.id}
              style={styles.conversationItem}
              onClick={() => onConversationSelect(conv.id)}
            >
              <div style={styles.convTitle}>{conv.title}</div>
              <div style={styles.convPreview}>{conv.last_message}</div>
              <div style={styles.convDate}>
                {new Date(conv.updated_at).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div style={styles.emptyState}>
          <p>No conversation history yet.</p>
          <p>Start a new conversation to see it here.</p>
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 15px',
    borderBottom: '1px solid #eee',
  },
  newButton: {
    padding: '6px 12px',
    backgroundColor: '#1976d2',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px',
  },
  historyList: {
    flex: 1,
    overflowY: 'auto',
    padding: '10px 0',
  },
  conversationItem: {
    padding: '12px 15px',
    borderBottom: '1px solid #eee',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  conversationItemHover: {
    backgroundColor: '#f5f5f5',
  },
  convTitle: {
    fontWeight: 'bold',
    marginBottom: '4px',
    color: '#333',
  },
  convPreview: {
    fontSize: '14px',
    color: '#666',
    marginBottom: '4px',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  convDate: {
    fontSize: '12px',
    color: '#999',
  },
  emptyState: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '20px',
    color: '#999',
    textAlign: 'center',
  },
};

export default ConversationHistory;