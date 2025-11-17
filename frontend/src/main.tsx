import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './styles/index.css';

// Apply saved theme early to avoid flicker
try {
  const saved = localStorage.getItem('aura:theme');
  if (saved === 'light') document.documentElement.classList.add('theme-light');
} catch {}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
