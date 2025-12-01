
import React, {useEffect, useState} from 'react';
import CompanyStock from './CompanyStock';

export default function MultiStock({apiBase="http://localhost:8000", defaultSymbols=[]}){
  const [symbols, setSymbols] = useState(defaultSymbols);
  const [input, setInput] = useState(defaultSymbols.join(', '));

  useEffect(()=>{
    setSymbols(defaultSymbols);
    setInput(defaultSymbols.join(', '));
  }, [defaultSymbols]);

  const apply = ()=>{
    const list = input.split(',').map(s=>s.trim().toUpperCase()).filter(s=>s);
    setSymbols(list);
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-4 flex gap-2 items-center">
        <input value={input} onChange={e=>setInput(e.target.value)} className="input flex-1" />
        <button onClick={apply} className="px-3 py-1 bg-indigo-600 text-white rounded">Track</button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {symbols.map(s => (
          <div key={s} className="bg-white dark:bg-gray-800 rounded p-3 shadow-sm">
            <CompanyStock apiBase={apiBase} symbol={s} />
          </div>
        ))}
      </div>
    </div>
  )
}
