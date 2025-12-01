# Nifty50 Predictor — FastAPI Backend

## Overview
This FastAPI backend serves two endpoints:
- `POST /predict` expects JSON `{open, high, low, close}` and returns `{predicted_next_close}`.
- `GET /history` returns test-set comparison if `data.csv` is present and model files are available.

The app will try to load `linear_reg_model.pkl` and `scaler.pkl` if they exist alongside the app.
If they are absent, the API uses a mock prediction function that simulates reasonable values.

## Setup (local)
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Files
- `app.py` — FastAPI app
- `requirements.txt`
- (optional) `linear_reg_model.pkl` and `scaler.pkl` to use the real model
- (optional) `data.csv` to populate `/history`

## Notes
- For production, use a process manager (gunicorn, uvicorn workers, or Docker) and secure CORS/origins.
