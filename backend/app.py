from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib, os
from typing import Optional
import pandas as pd

app = FastAPI(title='Nifty50 Predictor API (FastAPI)')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_reg_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
DATA_CSV = os.path.join(os.path.dirname(__file__), 'data.csv')

model = None
scaler = None

# Try to load model & scaler if present
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
except Exception as e:
    model = None
    scaler = None

class OHLC(BaseModel):
    open: float
    high: float
    low: float
    close: float

def mock_predict(close: float, history_closes=None):
    # Simple mock prediction: momentum + small noise
    if history_closes and len(history_closes) > 0:
        momentum = close - history_closes[-1]
        vol = max(1.0, (max(history_closes) - min(history_closes)) / max(1, len(history_closes)))
    else:
        momentum = 0.0
        vol = max(20.0, close * 0.002)
    noise = (np.random.rand() - 0.5) * vol * 2
    pred = close + 0.4 * momentum + noise
    return float(np.round(pred, 2))

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.post('/predict')
def predict(payload: OHLC):
    x = np.array([[payload.open, payload.high, payload.low, payload.close]], dtype=float)
    # If model available, use it
    if model is not None and scaler is not None:
        try:
            x_scaled = scaler.transform(x)
            p = model.predict(x_scaled)[0]
            return {'predicted_next_close': float(np.round(p, 2)), 'model': True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Model prediction failed: {e}')
    # else mock predict (optionally use data.csv if present)
    history_closes = None
    try:
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)
            if 'Close' in df.columns:
                history_closes = df['Close'].dropna().astype(float).tolist()[-50:]
    except Exception:
        history_closes = None
    pred = mock_predict(payload.close, history_closes)
    return {'predicted_next_close': pred, 'model': False}

@app.get('/history')
def history(limit: Optional[int] = 200):
    # If data.csv exists and model/scaler present, compute test-set preds similar to training split
    if os.path.exists(DATA_CSV):
        try:
            df = pd.read_csv(DATA_CSV)
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)
            for col in ['Open','High','Low','Close']:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','').str.strip(), errors='coerce')
            df['Target'] = df['Close'].shift(-1)
            df = df.dropna(subset=['Open','High','Low','Close','Target']).reset_index(drop=True)
            X = df[['Open','High','Low','Close']].copy()
            y = df['Target'].copy()
            split_idx = int(len(X) * 0.8)
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            # if model exists, predict
            preds = None
            if model is not None and scaler is not None:
                preds = model.predict(scaler.transform(X_test))
            rows = []
            for i in range(min(len(X_test), limit)):
                rows.append({
                    'date': df['Date'].iloc[split_idx + i].strftime('%Y-%m-%d'),
                    'open': float(X_test['Open'].iloc[i]),
                    'high': float(X_test['High'].iloc[i]),
                    'low': float(X_test['Low'].iloc[i]),
                    'close': float(X_test['Close'].iloc[i]),
                    'actual': float(y_test.iloc[i]),
                    'predicted': float(np.round(preds[i],2)) if preds is not None else None
                })
            metrics = {}
            if preds is not None:
                metrics['rmse'] = float(np.round(np.sqrt(np.mean((y_test.values - preds)**2)),4))
                metrics['n_test'] = int(len(y_test))
            return {'history': rows, 'metrics': metrics}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to prepare history: {e}')
    else:
        return {'history': [], 'metrics': {}}


# --- Added simulated company stock endpoints ---
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio, random, time
from collections import deque

# In-memory store for simulated stocks
_stock_state = {}
def _init_stock(symbol):
    if symbol in _stock_state:
        return
    # start price random around 50-500
    price = round(random.uniform(50, 500),2)
    history = deque(maxlen=240)  # keep last 240 ticks
    for i in range(60):
        price = round(price * (1 + random.uniform(-0.005,0.005)),2)
        history.append({'ts': time.time() - (60-i), 'price': price})
    _stock_state[symbol] = {'price': price, 'history': history, 'lock': asyncio.Lock()}

async def _tick_stock(symbol):
    """Simulate small random-walk update"""

    async with _stock_state[symbol]['lock']:
        price = _stock_state[symbol]['price']
        # random drift + volatility
        drift = random.uniform(-0.001,0.001)
        vol = random.uniform(-0.01,0.01)
        newp = round(max(0.1, price*(1+drift+vol)),2)
        _stock_state[symbol]['price'] = newp
        _stock_state[symbol]['history'].append({'ts': time.time(), 'price': newp})
        return newp

@app.get('/stock/{symbol}/current')
async def stock_current(symbol: str):
    symbol = symbol.upper()
    _init_stock(symbol)
    # tick once before returning to simulate live
    price = await _tick_stock(symbol)
    return {'symbol': symbol, 'price': price, 'timestamp': time.time()}

@app.get('/stock/{symbol}/predict')
async def stock_predict(symbol: str):
    """Simple next-day prediction using linear extrapolation on last N points"""
    symbol = symbol.upper()
    _init_stock(symbol)
    history = list(_stock_state[symbol]['history'])
    if len(history) < 3:
        raise HTTPException(status_code=400, detail='Not enough data to predict')
    # use linear regression (time vs price)
    xs = [h['ts'] for h in history]
    ys = [h['price'] for h in history]
    # normalize times
    x0 = xs[0]
    xs2 = [(x - x0) for x in xs]
    # calculate slope and intercept
    n = len(xs2)
    x_mean = sum(xs2)/n
    y_mean = sum(ys)/n
    num = sum((xs2[i]-x_mean)*(ys[i]-y_mean) for i in range(n))
    den = sum((xs2[i]-x_mean)**2 for i in range(n)) or 1.0
    slope = num/den
    intercept = y_mean - slope * x_mean
    # predict 24 hours ahead
    target = xs2[-1] + 24*3600
    pred = round(slope * target + intercept, 2)
    return {'symbol': symbol, 'predicted_next_day': float(pred), 'as_of': time.time()}

@app.get('/stock/{symbol}/history')
async def stock_history(symbol: str, limit: int = 60):
    symbol = symbol.upper()
    _init_stock(symbol)
    history = list(_stock_state[symbol]['history'])[-limit:]
    return {'symbol': symbol, 'history': history}

@app.get('/stock/{symbol}/stream')
async def stock_stream(symbol: str):
    """Server-Sent Events streaming of price updates - clients can connect with EventSource."""
    symbol = symbol.upper()
    _init_stock(symbol)
    async def event_generator():
        while True:
            try:
                price = await _tick_stock(symbol)
                data = f"data: {{\"symbol\": \"{symbol}\", \"price\": {price}, \"ts\": {time.time()}}}\n\n"
                yield data

                await asyncio.sleep(1.0)  # 1 second updates
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)
    return StreamingResponse(event_generator(), media_type='text/event-stream')

# --- end additions ---


# --- Yahoo Finance real-data endpoints (uses yfinance) ---
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

def fetch_current_price(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.history(period='1d', interval='1m')
        if info is not None and len(info)>0:
            latest = info['Close'].iloc[-1]
            return float(latest)
        data = t.info
        if 'currentPrice' in data:
            return float(data['currentPrice'])
    except Exception:
        pass
    raise HTTPException(status_code=500, detail='Failed to fetch price from yfinance')

def fetch_history(symbol, period='60d', interval='1d'):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval=interval)
        if hist is None or len(hist)==0:
            raise Exception('No history')
        out = []
        for idx,row in hist.iterrows():
            out.append({'date': str(idx.date()), 'open': float(row['Open']), 'high': float(row['High']),
                        'low': float(row['Low']), 'close': float(row['Close']), 'volume': int(row['Volume'])})
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to fetch history: {e}')

def predict_next_day(symbol, lookback_days=60):
    # Train a simple linear regression on closing prices over lookback_days and predict next close.
    try:
        hist = fetch_history(symbol, period=f"{lookback_days}d", interval='1d')
        if len(hist) < 5:
            raise HTTPException(status_code=400, detail='Not enough history for prediction')
        ys = np.array([h['close'] for h in hist]).reshape(-1,1)
        xs = np.arange(len(ys)).reshape(-1,1)
        model = LinearRegression()
        model.fit(xs, ys)
        next_x = np.array([[len(xs)]])
        pred = model.predict(next_x)[0][0]
        return float(np.round(pred,2))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

@app.get('/yf/{symbol}/current')
async def yf_current(symbol: str):
    symbol = symbol.upper()
    price = fetch_current_price(symbol)
    return {'symbol': symbol, 'price': price, 'timestamp': time.time()}

@app.get('/yf/{symbol}/history')
async def yf_history(symbol: str, period: str = '60d', interval: str = '1d'):
    symbol = symbol.upper()
    hist = fetch_history(symbol, period=period, interval=interval)
    return {'symbol': symbol, 'history': hist}

@app.get('/yf/{symbol}/predict')
async def yf_predict(symbol: str):
    symbol = symbol.upper()
    pred = predict_next_day(symbol)
    return {'symbol': symbol, 'predicted_next_day': pred, 'as_of': time.time()}

@app.get('/yf/bulk')
async def yf_bulk(symbols: str):
    """Accepts comma-separated symbols, returns current price and prediction for each."""
    syms = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    out = []
    for s in syms:
        try:
            price = fetch_current_price(s)
            pred = predict_next_day(s)
            out.append({'symbol': s, 'price': price, 'predicted_next_day': pred})
        except Exception as e:
            out.append({'symbol': s, 'error': str(e)})
    return {'results': out}

# --- end Yahoo Finance additions ---
