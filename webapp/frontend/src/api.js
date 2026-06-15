// Static databank client. The full input space (model × derivative × γ × δ) was
// precomputed by webapp/build_databank.py into JSON under /data, so the app needs
// no backend: every call snaps the sliders to the grid and reads a cached file.
const BASE = import.meta.env.BASE_URL + 'data/'

const _cache = new Map()
function load(name) {
  if (!_cache.has(name)) {
    _cache.set(name, fetch(BASE + name).then(r => {
      if (!r.ok) throw new Error(`${name} ${r.status}`)
      return r.json()
    }))
  }
  return _cache.get(name)
}

const manifest = load('manifest.json')

const nearest = (arr, x) => {
  let bi = 0, bd = Infinity
  for (let i = 0; i < arr.length; i++) { const d = Math.abs(arr[i] - x); if (d < bd) { bd = d; bi = i } }
  return bi
}

// snap (γ, δ) to the precomputed grid, respecting the convexity floor δ ≥ 3γ²/8
async function snap(gamma, delta) {
  const m = await manifest
  const gi = nearest(m.gamma, gamma)
  const floor = 3 * m.gamma[gi] * m.gamma[gi] / 8
  let di = nearest(m.delta, delta)
  if (m.delta[di] < floor) {
    di = m.delta.findIndex(d => d >= floor)
    if (di < 0) di = m.delta.length - 1
  }
  return { gi, di, key: `${gi}_${di}` }
}

export const getDerivatives = () => load('derivatives.json')

export const getPaths = req => load(`paths_${req.model}.json`)

async function config(req) {
  const file = await load(`${req.model}_${req.derivative}.json`)
  const { key } = await snap(req.gamma, req.delta)
  return { file, entry: file.entries[key] }
}

export async function getHedge(req) {
  const { file, entry } = await config(req)
  return { holdings: entry.theta.map(theta => ({ t: file.t, theta })) }
}

export async function getPnl(req) {
  const { file, entry } = await config(req)
  return {
    p0: file.p0, K: file.K, tau: entry.tau, complete: file.complete,
    hist: entry.hist, stats_mv: entry.stats_mv, stats_as: entry.stats_as,
  }
}

export async function getPortfolio(req) {
  const port = await load('portfolio.json')
  const { key } = await snap(req.gamma, req.delta)
  const e = port.entries[key]
  return { labels: port.labels, weights: port.weights, mv: e.mv, asym: e.asym }
}
