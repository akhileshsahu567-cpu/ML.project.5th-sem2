import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

export default function CompanyStock({ apiBase = "http://localhost:8000", symbol = "ACME" }) {
  const [price, setPrice] = useState(null);
  const [history, setHistory] = useState([]);
  const [prediction, setPrediction] = useState(null);

  // Load real history + prediction on first render
  useEffect(() => {
    // fetch REAL history from Yahoo Finance backend
    fetch(`${apiBase}/yf/${symbol}/history`)
      .then(r => r.json())
      .then(data => {
        if (data && data.history) {
          const processed = data.history.map(h => ({
            time: h.date,
            price: h.close
          }));
          setHistory(processed);
        }
      });

    // fetch REAL current price
    fetch(`${apiBase}/yf/${symbol}/current`)
      .then(r => r.json())
      .then(d => setPrice(d.price));

    // fetch REAL ML prediction
    fetch(`${apiBase}/yf/${symbol}/predict`)
      .then(r => r.json())
      .then(d => setPrediction(d.predicted_next_day));
  }, [apiBase, symbol]);

  // Poll real-time price every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${apiBase}/yf/${symbol}/current`)
        .then(r => r.json())
        .then(d => {
          setPrice(d.price);
        });
    }, 5000);

    return () => clearInterval(interval);
  }, [apiBase, symbol]);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold">Live - {symbol.toUpperCase()}</h2>
        <div className="text-right">
          <div className="text-sm text-gray-500">Current price</div>
          <div className="text-2xl font-bold">{price != null ? price.toFixed(2) : '--'}</div>
          <div className="text-sm text-gray-500">
            Predicted next day: {prediction != null ? prediction.toFixed(2) : '--'}
          </div>
        </div>
      </div>

      <div style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" minTickGap={20} />
            <YAxis domain={['auto', 'auto']} />
            <Tooltip />
            <Line type="monotone" dataKey="price" dot={false} stroke="#8884d8" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6">
        <p className="text-sm text-gray-600">
          This page displays real market data fetched live from Yahoo Finance,
          along with a real-time machine-learning prediction for the next day.
        </p>
      </div>
    </div>
  );
}

