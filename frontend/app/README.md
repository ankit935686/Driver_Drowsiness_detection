# Frontend App (Web + PWA)

This single frontend app serves both web and future PWA needs.

Current implementation:
- Browser camera preview using getUserMedia
- Start/stop detection controls
- Browser frame upload loop to backend `/detect/frame`
- Live runtime metrics (state, EAR, FPS, closed frames)
- Raw JSON status viewer for quick debugging
- PWA basics (`manifest.json`, `service-worker.js`)

How to run:
1. Start backend API: `python backend/src/api_server.py`
2. Open browser: `http://127.0.0.1:5000/`
