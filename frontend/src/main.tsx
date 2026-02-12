import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

console.log('[Main] Starting application...');

const rootElement = document.getElementById('root');
console.log('[Main] Root element:', rootElement);

if (rootElement) {
  createRoot(rootElement).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
  console.log('[Main] App rendered');
} else {
  console.error('[Main] Root element not found!');
}
