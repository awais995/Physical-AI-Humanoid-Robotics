import React from 'react';

const ResponseRenderer = ({ content, citations = [], confidence }) => {
  return (
    <div style={styles.container}>
      <div style={styles.content}>
        {content || ''}
      </div>

      {confidence !== undefined && (
        <div style={styles.confidenceContainer}>
          <div style={styles.confidenceBar}>
            <div
              style={{
                ...styles.confidenceFill,
                width: `${confidence * 100}%`,
                backgroundColor: confidence > 0.7 ? 'var(--ifm-color-success, #34a853)' :
                               confidence > 0.4 ? 'var(--ifm-color-warning, #f9ab00)' :
                               'var(--ifm-color-danger, #ea4335)'
              }}
            />
          </div>
          <div style={styles.confidenceText}>
            <strong>Confidence:</strong> {(confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {citations && citations.length > 0 && (
        <div style={styles.citationsSection}>
          <div style={styles.citationsHeader}>
            <h4 style={styles.citationsTitle}>Sources Referenced</h4>
            <div style={styles.citationsCount}>({citations.length} source{citations.length !== 1 ? 's' : ''})</div>
          </div>
          <div style={styles.citationsList}>
            {citations.map((citation, index) => (
              <div key={index} style={styles.citationItem}>
                <div style={styles.citationHeader}>
                  <div style={styles.citationSource}>
                    <span style={styles.sourceIcon}>📄</span>
                    <span style={styles.sourceName}>{citation.source}</span>
                  </div>
                  {(citation.chapter || citation.section || citation.page_number) && (
                    <div style={styles.citationLocation}>
                      {citation.chapter && <span>Ch. {citation.chapter}</span>}
                      {citation.section && <span>{citation.chapter ? ' · ' : ''}Sec. {citation.section}</span>}
                      {citation.page_number && <span>{(citation.chapter || citation.section) ? ' · ' : ''}Pg. {citation.page_number}</span>}
                    </div>
                  )}
                </div>
                <div style={styles.citationText}>
                  "{citation.text ? citation.text.substring(0, 120) : ''}{citation.text && citation.text.length > 120 ? '...' : ''}"
                </div>
              </div>
            ))}
          </div>
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
    lineHeight: '1.6',
    marginBottom: '12px'
  },
  confidenceContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '12px'
  },
  confidenceBar: {
    flex: 1,
    height: '6px',
    backgroundColor: 'var(--ifm-color-emphasis-200, #d0d0d0)',
    borderRadius: '3px',
    overflow: 'hidden'
  },
  confidenceFill: {
    height: '100%',
    transition: 'width 0.3s ease, background-color 0.3s ease'
  },
  confidenceText: {
    fontSize: '0.8rem',
    color: 'var(--ifm-color-emphasis-600, #666)',
    whiteSpace: 'nowrap'
  },
  citationsSection: {
    marginTop: '16px',
    paddingTop: '16px',
    borderTop: '1px solid var(--ifm-color-emphasis-300, #d0d0d0)'
  },
  citationsHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px'
  },
  citationsTitle: {
    margin: 0,
    fontSize: '0.95rem',
    fontWeight: '600',
    color: 'var(--ifm-heading-color, #242526)'
  },
  citationsCount: {
    fontSize: '0.8rem',
    color: 'var(--ifm-color-emphasis-600, #666)',
    backgroundColor: 'var(--ifm-color-emphasis-100, #f0f0f0)',
    padding: '2px 8px',
    borderRadius: '12px'
  },
  citationsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    margin: 0,
    padding: 0
  },
  citationItem: {
    padding: '12px',
    backgroundColor: 'var(--ifm-color-emphasis-100, #f8f9fa)',
    borderRadius: '8px',
    border: '1px solid var(--ifm-color-emphasis-200, #e0e0e0)',
    borderLeft: '3px solid var(--ifm-color-primary, #3578e5)'
  },
  citationHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '6px',
    flexWrap: 'wrap',
    gap: '8px'
  },
  citationSource: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    flex: 1
  },
  sourceIcon: {
    fontSize: '0.9rem'
  },
  sourceName: {
    fontWeight: '500',
    fontSize: '0.9rem',
    color: 'var(--ifm-heading-color, #242526)'
  },
  citationLocation: {
    fontSize: '0.8rem',
    color: 'var(--ifm-color-emphasis-600, #666)',
    display: 'flex',
    gap: '4px',
    flex: '0 0 auto'
  },
  citationText: {
    fontStyle: 'italic',
    color: 'var(--ifm-color-emphasis-700, #444950)',
    fontSize: '0.9rem',
    lineHeight: '1.4',
    borderLeft: '2px solid var(--ifm-color-emphasis-300, #d0d0d0)',
    paddingLeft: '12px',
    marginLeft: '4px'
  }
};

export default ResponseRenderer;