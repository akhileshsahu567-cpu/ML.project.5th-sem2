
import React, {useState} from 'react'
import Nifty50Predictor from './components/Nifty50Predictor'
import MultiStock from './components/MultiStock'

export default function App(){
  const [page, setPage] = useState('nifty');
  const apiBase="http://localhost:8000";
  const defaults = ['RELIANCE.NS','TCS.NS','INFY.NS','HDFCBANK.NS','ITC.NS'];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="p-4 bg-white dark:bg-gray-800 shadow-sm">
        <nav className="max-w-6xl mx-auto flex gap-4">
          <button onClick={()=>setPage('nifty')} className={`px-3 py-1 rounded ${page==='nifty'?'bg-indigo-600 text-white':''}`}>Nifty Predictor</button>
          <button onClick={()=>setPage('multi')} className={`px-3 py-1 rounded ${page==='multi'?'bg-indigo-600 text-white':''}`}>Multiple Stocks</button>
          <div className="ml-auto text-sm text-gray-500">API: {apiBase}</div>
        </nav>
      </header>
      <main className="py-6">
        {page==='nifty' ? <Nifty50Predictor apiBase={apiBase} /> : <MultiStock apiBase={apiBase} defaultSymbols={defaults} />}
      </main>
    </div>
  )
}
