// Thin client over the FastAPI backend. Every number is a real sigcore call.
const BASE = import.meta.env.VITE_API ?? '/api'

async function post(path, body) {
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path} ${res.status}`)
  return res.json()
}

export const getDerivatives = () => fetch(BASE + '/derivatives').then(r => r.json())
export const getPaths = req => post('/paths', req)
export const getHedge = req => post('/hedge', req)
export const getPnl = req => post('/pnl', req)
export const getPortfolio = req => post('/portfolio', req)
