import React from 'react';

const QueryModeSelector = ({ selectedMode, onModeChange }) => {
  const modes = [
    { value: 'global', label: 'Global Book', description: 'Search entire book content' },
    { value: 'selected-text-only', label: 'Selected Text Only', description: 'Only answer from selected text' }
  ];

  return (
    <div style={styles.container}>
      <label style={styles.label}>Query Mode:</label>
      <div style={styles.modeSelector}>
        {modes.map((mode) => (
          <label key={mode.value} style={styles.radioLabel}>
            <input
              type="radio"
              name="query-mode"
              value={mode.value}
              checked={selectedMode === mode.value}
              onChange={(e) => onModeChange(e.target.value)}
              style={styles.radioInput}
            />
            <span style={styles.radioButton}>
              <span style={styles.modeLabel}>{mode.label}</span>
              <span style={styles.modeDescription}>{mode.description}</span>
            </span>
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
  label: {
    fontSize: '0.9em',
    marginBottom: '5px',
    color: '#666'
  },
  modeSelector: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    cursor: 'pointer',
    fontSize: '0.9em'
  },
  radioInput: {
    display: 'none'
  },
  radioButton: {
    display: 'flex',
    flexDirection: 'column',
    padding: '6px 10px',
    border: '2px solid transparent',
    borderRadius: '6px',
    transition: 'border-color 0.2s',
    cursor: 'pointer'
  },
  radioButtonActive: {
    borderColor: '#1976d2',
    backgroundColor: '#e3f2fd'
  },
  modeLabel: {
    fontWeight: 'bold',
    fontSize: '1em',
    marginBottom: '2px'
  },
  modeDescription: {
    fontSize: '0.85em',
    color: '#666'
  }
};

export default QueryModeSelector;