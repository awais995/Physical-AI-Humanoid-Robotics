import React from 'react';

const QueryModeSelector = ({ selectedMode, onModeChange }) => {
  const modes = [
    { value: 'global', label: 'Global Book Search', description: 'Search entire book content', icon: '🔍' },
    { value: 'selected-text-only', label: 'Selected Text Only', description: 'Only answer from selected text', icon: '📝' }
  ];

  return (
    <div style={styles.container}>
      <div style={styles.modeSelector}>
        {modes.map((mode) => (
          <label
            key={mode.value}
            style={{
              ...styles.radioButton,
              ...(selectedMode === mode.value ? styles.radioButtonActive : {})
            }}
          >
            <input
              type="radio"
              name="query-mode"
              value={mode.value}
              checked={selectedMode === mode.value}
              onChange={(e) => onModeChange(e.target.value)}
              style={styles.radioInput}
            />
            <div style={styles.radioButtonContent}>
              <div style={styles.radioButtonIcon}>{mode.icon}</div>
              <div style={styles.radioButtonText}>
                <div style={styles.modeLabel}>{mode.label}</div>
                <div style={styles.modeDescription}>{mode.description}</div>
              </div>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end'
  },
  modeSelector: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    width: '240px'  // Fixed width for consistency
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '0.9em',
    padding: '10px',
    borderRadius: '8px',
    border: '2px solid transparent',
    transition: 'all 0.2s ease'
  },
  radioInput: {
    display: 'none'
  },
  radioButton: {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '0.9em',
    padding: '12px',
    borderRadius: '8px',
    border: '2px solid var(--ifm-color-emphasis-200, #d0d0d0)',
    backgroundColor: 'var(--ifm-background-color, #ffffff)',
    transition: 'all 0.2s ease',
    gap: '10px'
  },
  radioButtonActive: {
    borderColor: 'var(--ifm-color-primary, #3578e5)',
    backgroundColor: 'var(--ifm-color-primary-background, #e6f0fa)',
    boxShadow: '0 0 0 3px rgba(53, 120, 229, 0.1)'
  },
  radioButtonContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flex: 1
  },
  radioButtonIcon: {
    fontSize: '1.2rem',
    minWidth: '24px',
    textAlign: 'center'
  },
  radioButtonText: {
    flex: 1
  },
  modeLabel: {
    fontWeight: '600',
    fontSize: '0.95rem',
    color: 'var(--ifm-font-color-base, #242526)',
    marginBottom: '2px'
  },
  modeDescription: {
    fontSize: '0.85rem',
    color: 'var(--ifm-color-emphasis-600, #666)',
    lineHeight: '1.3'
  }
};

export default QueryModeSelector;