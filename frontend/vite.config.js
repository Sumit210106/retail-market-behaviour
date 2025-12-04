import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'


// https://vite.dev/config/
export default defineConfig({
  plugins: [react(),tailwindcss()],
})




// Edit
// Included Paths
// Changes that match these paths will trigger a new build.


// Add Included Path
// Ignored Paths
// Changes that match these paths will not trigger a new build.


// Add Ignored Path
// Build Command
// Render runs this command to build your app before each deploy.
// backend/ $
// pip install -r requirements.txt

// Edit
// Pre-Deploy CommandOptional
// Render runs this command before the start command. Useful for database migrations and static asset uploads.
// backend/ $

// Edit
// Start Command
// Render runs this command to start your app with each deploy.
// backend/ $
// uvicorn app:app --host 0.0.0.0 --port 10000

