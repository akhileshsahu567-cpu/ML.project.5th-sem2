import React, {useEffect, useState, useMemo} from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, Area } from 'recharts';
import { Sun, Moon, Database, Play, Download, GitBranch, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

function movingAverage(data, key='close', period=5){
  const out = [];
  for(let i=0;i<data.length;i++){
    const start = Math.max(0, i-period+1);
    const slice = data.slice(start, i+1).map(d => Number(d[key]) || 0);
    const avg = slice.reduce((a,b)=>a+b,0)/slice.length;
    out.push(Number(avg.toFixed(2)));
  }
  return out;
}

export default function Nifty50Predictor({ apiBase = '' }) {
  const [theme, setTheme] = useState('light');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState({open:'', high:'', low:'', close:''});
  const [prediction, setPrediction] = useState(null);
  const [rmse, setRmse] = useState(null);
  const [message, setMessage] = useState('');
  const [mockMode, setMockMode] = useState(true);

  useEffect(()=>{ document.documentElement.classList.toggle('dark', theme==='dark'); },[theme]);

  // Load small embedded demo history for UI (so charts have content even with no backend)
  useEffect(()=>{
    // generate a small demo history if no /history endpoint available
    async function loadHistory(){
      try{
        const res = await fetch(`${apiBase}/history`);
        if(res.ok){
          const j = await res.json();
          setHistory(j.slice(-200));
          if(j.metrics) setRmse(j.metrics.rmse || null);
          return;
        }
      }catch(e){ /* ignore */ }

      // fallback demo data (synthetic recent trend)
      const demo = [];
      let base = 24000;
      for(let i=0;i<120;i++){
        const date = new Date();
        date.setDate(date.getDate() - (120 - i));
        const open = base + Math.sin(i/5)*150 + (Math.random()-0.5)*30;
        const close = open + (Math.random()-0.5)*80;
        const high = Math.max(open, close) + Math.random()*40;
        const low = Math.min(open, close) - Math.random()*40;
        demo.push({date: date.toISOString().slice(0,10), open: Math.round(open*100)/100, high: Math.round(high*100)/100, low: Math.round(low*100)/100, close: Math.round(close*100)/100, actual: null, predicted: null});
        base += 2; // gentle uptrend
      }
      setHistory(demo);
    }
    loadHistory();
  },[apiBase]);

  // Derived chart data with moving average
  const chartData = useMemo(()=>{
    const data = history.map(h => ({ date: h.date, actual: h.actual, predicted: h.predicted, close: h.close }));
    const ma = movingAverage(data, 'close', 10);
    return data.map((d,i)=>({ ...d, ma: ma[i] }));
  }, [history]);

  useEffect(()=>{
    // compute a demo RMSE if in mock mode
    if(mockMode && history.length>0){
      const preds = history.filter(h=>h.predicted!=null).map(h=>h.predicted);
      if(preds.length>0){
        const acts = history.filter(h=>h.actual!=null).map(h=>h.actual);
        if(acts.length===preds.length && acts.length>0){
          const mse = acts.reduce((s,a,idx)=> s + Math.pow(a - preds[idx],2), 0) / acts.length;
          setRmse(Math.sqrt(mse));
        }
      }
    }
  }, [mockMode, history]);

  function handleChange(e){ const {name, value} = e.target; setInput(prev=>({...prev, [name]: value})); }

  function fillWithLast(){ if(history.length===0){ setMessage('No history loaded to autofill.'); return; } const last = history[history.length-1]; setInput({open:last.open, high:last.high, low:last.low, close:last.close}); setMessage('Autofilled with last available row.'); }

  // Mock prediction generator using recent volatility
  function mockPredict(open, high, low, close){
    // estimate volatility from recent closes
    const closes = history.slice(-20).map(h=>Number(h.close)).filter(Boolean);
    const vol = closes.length>1 ? (Math.max(...closes)-Math.min(...closes))/Math.max(1,closes.length) : Math.max(20, Math.abs(close)*0.002);
    const noise = (Math.random()-0.5) * vol * 2;
    // simple model: next = close + momentum + noise
    const momentum = (closes.length>0 ? (close - closes[closes.length-1]) : 0);
    const pred = close + momentum*0.4 + noise;
    return Math.round(pred*100)/100;
  }

  async function runPredict(){ setMessage(''); const nums = ['open','high','low','close'].map(k=>parseFloat(input[k])); if(nums.some(x=>Number.isNaN(x))){ setMessage('Please enter valid numeric OHLC values.'); return; } setLoading(true); try{
      if(mockMode || !apiBase){
        // simulate latency
        await new Promise(r=>setTimeout(r, 400 + Math.random()*600));
        const pred = mockPredict(nums[0], nums[1], nums[2], nums[3]);
        setPrediction(pred);
        // append to local history
        setHistory(prev=>[...prev, {date: new Date().toISOString().slice(0,10), open:nums[0], high:nums[1], low:nums[2], close:nums[3], actual:null, predicted: pred}].slice(-500));
        setMessage('Mock prediction complete');
      } else {
        const res = await fetch(`${apiBase}/predict`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({open: nums[0], high: nums[1], low: nums[2], close: nums[3]}) });
        if(!res.ok) throw new Error('Prediction failed');
        const j = await res.json();
        const p = j.predicted_next_close ?? j.predicted_nextClose ?? j.predicted ?? null;
        setPrediction(p);
        setHistory(prev=>[...prev, {date: new Date().toISOString().slice(0,10), open:nums[0], high:nums[1], low:nums[2], close:nums[3], actual:null, predicted: p}].slice(-500));
        setMessage('Prediction complete');
      }
    }catch(err){ console.error(err); setMessage('Prediction request failed — check backend.'); }finally{ setLoading(false); } }

  function exportCSV(){ const rows = ['date,open,high,low,close,actual,predicted', ...history.map(r=>`${r.date || ''},${r.open},${r.high},${r.low},${r.close},${r.actual || ''},${r.predicted || ''}`)]; const blob = new Blob([rows.join('\n')], {type:'text/csv'}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'predictions_history.csv'; a.click(); URL.revokeObjectURL(url); }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-100 transition-colors">
      <div className="max-w-6xl mx-auto p-6">
        <header className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="rounded-2xl p-3 bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-lg">
              <Database size={28} />
            </div>
            <div>
              <h1 className="text-2xl font-bold">NIFTY 50 Predictor</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">Next-day close prediction • Demo frontend</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
              <Zap size={16} /> <span className="font-medium">Mode:</span>
              <span className="ml-1 font-semibold">{mockMode? 'Mock' : 'Live'}</span>
            </div>

            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={mockMode} onChange={e=>setMockMode(e.target.checked)} />
                Mock Mode
              </label>

              <button onClick={()=>setTheme(prev=>prev==='light'?'dark':'light')} className="p-2 rounded-lg border dark:border-gray-700">
                {theme==='light' ? <Moon size={18} /> : <Sun size={18} />}
              </button>
            </div>
          </div>
        </header>

        <main className="grid grid-cols-12 gap-6">
          <section className="col-span-12 lg:col-span-7 bg-white dark:bg-gray-800 p-5 rounded-2xl shadow">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Price Chart (Close + MA)</h2>
              <div className="flex gap-2">
                <button onClick={()=>setHistory([])} className="px-3 py-1 rounded bg-red-50 dark:bg-red-900 text-red-600 dark:text-red-200 text-sm">Clear</button>
                <button onClick={exportCSV} className="px-3 py-1 rounded bg-blue-600 text-white text-sm flex items-center gap-2"><Download size={14}/> Export CSV</button>
              </div>
            </div>

            <div style={{height:420}}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{top:10,right:20,left:0,bottom:0}}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" minTickGap={20} />
                  <YAxis domain={['dataMin - 200', 'dataMax + 200']} />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="close" fillOpacity={0.06} stroke={null} />
                  <Line type="monotone" dataKey="close" stroke="#10b981" dot={false} name="Close" strokeWidth={2} />
                  <Line type="monotone" dataKey="ma" stroke="#6366f1" dot={false} name="MA(10)" strokeWidth={2} strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="predicted" stroke="#f97316" dot={{r:2}} name="Predicted" strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <aside className="col-span-12 lg:col-span-5 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-5 rounded-2xl shadow">
              <h3 className="font-semibold mb-2">Predict Next-day Close</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">Enter today's OHLC values or autofill from history.</p>

              <div className="grid grid-cols-2 gap-3">
                <input name="open" value={input.open} onChange={handleChange} placeholder="Open" className="input" />
                <input name="high" value={input.high} onChange={handleChange} placeholder="High" className="input" />
                <input name="low" value={input.low} onChange={handleChange} placeholder="Low" className="input" />
                <input name="close" value={input.close} onChange={handleChange} placeholder="Close" className="input" />
              </div>

              <div className="flex items-center gap-3 mt-4">
                <button onClick={fillWithLast} className="px-4 py-2 rounded-lg border dark:border-gray-700 text-sm">Autofill</button>
                <motion.button onClick={runPredict} whileTap={{scale:0.98}} className="px-4 py-2 rounded-lg bg-indigo-600 text-white flex items-center gap-2">{loading ? 'Running...' : 'Predict'}</motion.button>
                <button onClick={()=>{ setInput({open:'',high:'',low:'',close:''}); setPrediction(null); setMessage(''); }} className="px-3 py-2 rounded-lg border text-sm">Reset</button>
              </div>

              <div className="mt-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Prediction</div>
                <div className="mt-1 text-2xl font-bold">{prediction ? prediction.toFixed(2) : '—'}</div>
                {message && <div className="mt-2 text-sm text-yellow-600 dark:text-yellow-300">{message}</div>}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-2xl shadow">
              <h4 className="font-semibold">Recent Predictions / History</h4>
              <div className="mt-3 max-h-64 overflow-auto">
                <table className="w-full text-sm table-fixed">
                  <thead className="text-gray-500">
                    <tr>
                      <th className="w-1/4 text-left">Date</th>
                      <th className="w-1/4">Close</th>
                      <th className="w-1/4">Pred</th>
                      <th className="w-1/4">Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.slice(-50).reverse().map((r,i)=> (
                      <tr key={i} className="border-t border-gray-100 dark:border-gray-700">
                        <td className="py-2 text-left">{r.date}</td>
                        <td className="py-2">{r.close}</td>
                        <td className="py-2">{r.predicted ?? '—'}</td>
                        <td className="py-2 text-xs text-gray-500">{r.actual ? 'Actual available' : 'Live / predicted'}</td>
                      </tr>
                    ))}
                    {history.length===0 && <tr><td colSpan={4} className="py-4 text-center text-gray-400">No history loaded — connect /history endpoint or run a few predictions.</td></tr>}
                  </tbody>
                </table>
              </div>

              <div className="mt-3 flex gap-2">
                <button onClick={exportCSV} className="px-3 py-2 rounded bg-green-600 text-white text-sm">Export CSV</button>
                <button onClick={()=>window.location.reload()} className="px-3 py-2 rounded border text-sm">Reload</button>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-4 rounded-2xl shadow text-sm">
              <h5 className="font-semibold">Notes & Tips</h5>
              <ul className="mt-2 list-disc ml-5 text-gray-500">
                <li>Toggle <strong>Mock Mode</strong> to demo the UI without a backend.</li>
                <li>To connect a real backend, set <code>apiBase</code> in <code>src/App.jsx</code>.</li>
                <li>For better accuracy, add lag features and upgrade model server-side (RandomForest / XGBoost).</li>
              </ul>
            </div>

          </aside>
        </main>
      </div>

      <style>{` .input { padding: 0.6rem 0.75rem; border-radius: 0.6rem; border: 1px solid rgba(0,0,0,0.06); background: transparent; } .input:focus { outline: 2px solid rgba(99,102,241,0.18); } `}</style>
    </div>
  );
}
