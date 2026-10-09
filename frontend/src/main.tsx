import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { GAME_NAME } from './config'
import './index.css'
import App from './App.tsx'

// index.html's <title> is only the pre-JS default; the real name lives in
// one variable (config.ts: GAME_NAME), so take over the tab title here.
document.title = GAME_NAME

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
