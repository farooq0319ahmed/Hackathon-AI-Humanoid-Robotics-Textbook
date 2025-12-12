import React from 'react';
import clsx from 'clsx';
import styles from './Exercise.module.css';

const ExerciseIcon = () => (
  <svg
    className={styles.exerciseIcon}
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    width="24"
    height="24"
  >
    <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
  </svg>
);

export default function Exercise({title, children, difficulty = "intermediate"}) {
  const difficultyColors = {
    beginner: {bg: '#e8f5e9', border: '#4caf50'},
    intermediate: {bg: '#fff8e1', border: '#ff9800'},
    advanced: {bg: '#ffebee', border: '#f44336'}
  };

  const colors = difficultyColors[difficulty] || difficultyColors.intermediate;

  return (
    <div
      className="exercise-box"
      style={{
        backgroundColor: colors.bg,
        borderLeft: `4px solid ${colors.border}`
      }}
    >
      <div style={{display: 'flex', alignItems: 'center', marginBottom: '12px'}}>
        <ExerciseIcon />
        <h3 style={{margin: 0, marginLeft: '8px'}}>{title || "Hands-On Exercise"}</h3>
        <span style={{
          marginLeft: 'auto',
          padding: '2px 8px',
          borderRadius: '12px',
          backgroundColor: colors.border,
          color: 'white',
          fontSize: '0.8em'
        }}>
          {difficulty}
        </span>
      </div>
      <div>{children}</div>
    </div>
  );
}