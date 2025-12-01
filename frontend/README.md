# Nifty50 Predictor — Frontend (Enhanced)

This enhanced frontend includes:
- Polished UI and animations
- Improved charts (moving average, area under curve)
- Mock mode (simulate /predict locally so the UI works offline)
- Instructions for deployment (Vercel) and later connecting a real backend

## Setup (VS Code)

1. Open the folder in VS Code.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start dev server:
   ```bash
   npm run dev
   ```
   Vite will show the local URL (e.g. http://localhost:5173).

4. The app has a **Mock Mode** switch in the top-right. Turn it ON to simulate predictions locally (no backend required).

## Mock Mode
When Mock Mode is **ON**, the frontend simulates predictions using a simple noise model based on recent volatility and shows RMSE sample metrics. This is for demo/testing only.

## Deploy to Vercel (static frontend)
1. Commit the project to a GitHub repo.
2. Sign in to Vercel and import the repo.
3. Set the build command to `npm run build` and the output directory to `dist`.
4. If you later have a backend, set `apiBase` in `src/App.jsx` to the deployed backend URL.

## Connect Real Backend Later
- The frontend calls `POST /predict` with JSON `{open, high, low, close}` and expects `{predicted_next_close}`.
- To switch from mock to real backend, set `apiBase` in `src/App.jsx` to your backend base URL (e.g., `https://api.myapp.com`).

## Notes
- This is frontend-only. Mock mode is provided so you can demo the UI without a server.
