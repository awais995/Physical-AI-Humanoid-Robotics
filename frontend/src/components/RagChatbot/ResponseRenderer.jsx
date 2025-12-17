import React from 'react';

const ResponseRenderer = ({ content, citations = [], confidence }) => {
  return (
    <div style={styles.container}>
      <div style={styles.content}>
        {content}
      </div>

      {citations && citations.length > 0 && (
        <div style={styles.citationsSection}>
          <h4 style={styles.citationsTitle}>Citations:</h4>
          <ul style={styles.citationsList}>
            {citations.map((citation, index) => (
              <li key={index} style={styles.citationItem}>
                <div style={styles.citationSource}>
                  <strong>Source:</strong> {citation.source}
                  {citation.chapter && <span>, Chapter: {citation.chapter}</span>}
                  {citation.section && <span>, Section: {citation.section}</span>}
                  {citation.page_number && <span>, Page: {citation.page_number}</span>}
                </div>
                <div style={styles.citationText}>
                  "{citation.text}"
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {confidence !== undefined && (
        <div style={styles.confidenceIndicator}>
          <strong>Confidence:</strong> {(confidence * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    width: '100%'
  },
  content: {
    marginBottom: '10px',
    lineHeight: '1.5'
  },
  citationsSection: {
    marginTop: '10px',
    paddingTop: '10px',
    borderTop: '1px solid #ddd',
    fontSize: '0.9em'
  },
  citationsTitle: {
    margin: '0 0 8px 0',
    fontSize: '1em',
    fontWeight: 'bold',
    color: '#555'
  },
  citationsList: {
    margin: 0,
    padding: '0 0 0 20px',
    listStyle: 'none'
  },
  citationItem: {
    marginBottom: '10px',
    padding: '8px',
    backgroundColor: '#f9f9f9',
    borderRadius: '4px',
    borderLeft: '3px solid #1976d2'
  },
  citationSource: {
    marginBottom: '4px',
    fontWeight: '500',
    color: '#333'
  },
  citationText: {
    fontStyle: 'italic',
    color: '#666',
    fontSize: '0.95em'
  },
  confidenceIndicator: {
    marginTop: '8px',
    fontSize: '0.85em',
    color: '#666',
    fontStyle: 'italic'
  }
};

export default ResponseRenderer;