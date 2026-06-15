import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from GitHub Pages under /SignatureTrading/. The app is fully static —
// it reads the precomputed databank in public/data, so there is no backend proxy.
export default defineConfig({
  base: '/SignatureTrading/',
  plugins: [react()],
})
