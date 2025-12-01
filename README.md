 # ML.project.5th-sem2

ML.project.5th-sem — Combined Frontend + FastAPI Backend

## Folder structure:
- frontend/ → React + Vite + Tailwind (mock mode enabled)
- backend/ → FastAPI backend (app.py, requirements.txt, Dockerfile)
- README.md → This file

Frontend expects:
- POST /predict  
- GET /history  

Backend (FastAPI) supports these and loads model files  
`linear_reg_model.pkl` and `scaler.pkl` from the backend/ folder if available.

---

## How to run:

### 1) Frontend:
```bash
cd frontend
npm install
npm run dev

