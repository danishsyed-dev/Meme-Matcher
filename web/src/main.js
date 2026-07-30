import './styles/reset.css';
import './styles/tokens.css';
import './styles/main.css';
import { initApp } from './ui/app.js';

initApp().catch(err => {
  console.error('Failed to initialize Meme Matcher:', err);
  const el = document.getElementById('loading-message');
  if (el) {
    el.textContent = `Error: ${err.message}`;
  }
});
