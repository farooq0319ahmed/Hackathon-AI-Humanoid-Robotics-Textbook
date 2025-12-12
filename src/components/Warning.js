import React from 'react';
import clsx from 'clsx';
import styles from './Warning.module.css';

const WarningIcon = () => (
  <svg
    className={styles.warningIcon}
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    width="24"
    height="24"
  >
    <path fill="currentColor" d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
  </svg>
);

export default function Warning({title, children, type = "warning"}) {
  const typeColors = {
    warning: {bg: '#fff8f8', border: '#ff6b6b', icon: '#ff6b6b'},
    info: {bg: '#f0f8ff', border: '#4a90e2', icon: '#4a90e2'},
    tip: {bg: '#f0fff0', border: '#51cf66', icon: '#51cf66'}
  };

  const colors = typeColors[type] || typeColors.warning;

  return (
    <div
      className="warning-box"
      style={{
        backgroundColor: colors.bg,
        borderLeft: `4px solid ${colors.border}`,
        border: `1px solid ${colors.border}`,
        borderRadius: '4px'
      }}
    >
      <div style={{display: 'flex', alignItems: 'center', marginBottom: '8px'}}>
        <WarningIcon style={{fill: colors.icon}} />
        <h3 style={{margin: 0, marginLeft: '8px', color: colors.border}}>
          {title || type.charAt(0).toUpperCase() + type.slice(1)}
        </h3>
      </div>
      <div>{children}</div>
    </div>
  );
}