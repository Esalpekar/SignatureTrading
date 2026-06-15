import React, { useEffect, useMemo, useState } from 'react'
import { getDerivatives, getPaths, getHedge, getPnl, getPortfolio } from './api'
import { PnlHistogram, PolyPlot, Holdings, PricePaths, Simplex } from './charts'

const HESTON = { x0: 1.0, v0: 0.04, r: 0.0, kappa: 2.0, theta: 0.04, xi: 0.6, rho: -0.8, T: 1.0 }
const GBM = { x0: 1.0, r: 0.0, sigma: 0.2, T: 1.0 }
const DEPTH = 1  

function useDebounced(value, ms) {
  const [v, setV] = useState(value)
  useEffect(() => { const id = setTimeout(() => setV(value), ms); return () => clearTimeout(id) }, [value, ms])
  return v
}

const Num = ({ children }) => <span className="font-mono tabular-nums">{children}</span>
const fmt = (x, d = 4) => (x == null ? '.' : Number(x).toFixed(d))

function Stat({ label, mv, as }) {
  return (
    <div className="flex items-baseline justify-between py-1 border-b border-divider">
      <span className="text-slate text-xs uppercase tracking-wide">{label}</span>
      <span className="font-mono text-sm"><span className="text-slate">{fmt(mv)}</span> → <span className="text-ink font-semibold">{fmt(as)}</span></span>
    </div>
  )
}

export default function App() {
  const [derivs, setDerivs] = useState({})
  const [model, setModel] = useState('heston')
  const [derivative, setDerivative] = useState('asian')
  const [gamma, setGamma] = useState(10)
  const [delta, setDelta] = useState(40)
  const [paths, setPathsData] = useState(null)
  const [hedge, setHedge] = useState(null)
  const [pnl, setPnl] = useState(null)
  const [portfolio, setPortfolio] = useState(null)
  const [loading, setLoading] = useState(false)

  const params = model === 'heston' ? HESTON : GBM
  const rebalance_steps = model === 'gbm' ? 4 : 3  

  useEffect(() => { getDerivatives().then(setDerivs).catch(() => {}) }, [])

  // convexity floor: delta >= 3 gamma^2 / 8
  const deltaMin = (3 * gamma * gamma) / 8
  useEffect(() => { if (delta < deltaMin) setDelta(deltaMin) }, [gamma]) // eslint-disable-line

  const req = useMemo(() => ({
    model, params, derivative, gamma, delta, depth: DEPTH, rebalance_steps, n_paths: 6000,
  }), [model, derivative, gamma, delta])
  const dreq = useDebounced(req, 250)

  useEffect(() => { getPaths({ model, params }).then(setPathsData).catch(() => {}) }, [model])

  useEffect(() => {
    let alive = true
    setLoading(true)
    Promise.all([getHedge(dreq), getPnl(dreq)]).then(([h, p]) => {
      if (!alive) return
      setHedge(h); setPnl(p); setLoading(false)
    }).catch(() => alive && setLoading(false))
    return () => { alive = false }
  }, [dreq])

  // the 3-asset portfolio responds to the risk profile only (fixed example)
  useEffect(() => {
    let alive = true
    getPortfolio(dreq).then(p => alive && setPortfolio(p)).catch(() => {})
    return () => { alive = false }
  }, [dreq.gamma, dreq.delta])

  const d = derivs[derivative] || {}

  return (
    <div className="min-h-screen bg-paper py-6 px-4">
     <div className="max-w-6xl mx-auto border border-ink px-6 py-8">
      <header className="mb-8 border-b border-ink pb-4">
        <h1 className="text-2xl font-semibold tracking-tight">Signature-theoretic portfolio hedging of exotic derivatives under a personalized risk profile</h1>
        <p className="font-serif text-slate mt-1 max-w-2xl">
          Pick a world, a contract, and a risk attitude. Watch the optimal hedge and how your
          attitude to loss reshapes the tail of the shortfall.
        </p>
      </header>

      <div className="grid md:grid-cols-[320px_1fr] gap-8">
        {/* ---- pipeline controls ---- */}
        <div className="space-y-6">
          <Step n="1" title="Model of the underlying" hint="the assumed behaviour of the asset price over time.">
            <div className="flex gap-2">
              {['heston', 'gbm'].map(m => (
                <button key={m} onClick={() => setModel(m)}
                  className={`px-3 py-1 text-sm border ${model === m ? 'bg-ink text-paper border-ink' : 'border-grid text-slate'}`}>
                  {m === 'heston' ? 'Heston' : 'GBM'}
                </button>
              ))}
            </div>
            <PricePaths data={paths} />
            <p className="text-xs text-slate">{model === 'gbm'
              ? 'Standard Geometric Brownian Motion.'
              : 'A model with stochastic volatility.'}</p>
          </Step>

          <Step n="2" title="Derivative" hint="the contract you are hedging. What you owe at maturity.">
            <div className="flex flex-col gap-1">
              {Object.entries(derivs).map(([k, v]) => (
                <button key={k} onClick={() => setDerivative(k)}
                  className={`text-left px-3 py-2 border ${derivative === k ? 'border-ink' : 'border-divider'}`}>
                  <div className="text-sm font-medium">{v.label}</div>
                  <div className="text-xs text-slate font-serif">{v.blurb}</div>
                </button>
              ))}
            </div>
            <div className="mt-2 p-2 bg-fill text-xs">
              <div className="text-slate mb-1">covector f (level {d.level}):</div>
              <div className="font-mono break-words">{d.covector}</div>
            </div>
            <p className="text-xs text-slate font-serif">
              These presets are a subset of the available contracts. The same method hedges any
              continuous function of the stock's price path.
            </p>
          </Step>

          <Step n="3" title="Loss polynomial" hint="how you weigh losses of different sizes. The cubic tilts the penalty toward losses. You fear a big loss more than you value an equal gain.">
            <PolyPlot gamma={gamma} delta={delta} />
            <Slider label="γ (downside tilt)" value={gamma} min={0} max={30} step={0.5} onChange={setGamma} />
            <Slider label="δ (tail weight)" value={delta} min={deltaMin} max={400} step={1} onChange={setDelta} />
            <p className="text-xs text-slate">convex region enforced: δ ≥ 3γ²/8 = <Num>{fmt(deltaMin, 2)}</Num></p>
          </Step>
        </div>

        {/* ---- result panel ---- */}
        <div className="space-y-6">
          <div className="flex items-baseline justify-between">
            <h2 className="text-lg font-semibold">Result {loading && <span className="text-slate text-sm font-normal">· computing…</span>}</h2>
            <div className="font-mono text-sm text-slate">fair price p₀ = <span className="text-ink">{fmt(pnl?.p0)}</span> · K = {fmt(pnl?.K)}</div>
          </div>

          {pnl?.complete && (
            <div className="text-xs border border-ink p-2">
              This configuration is a complete market. The hedge replicates exactly, so the risk
              profile cannot move the tail. Switch to Heston or a path-dependent contract.
            </div>
          )}

          <section>
            <h3 className="text-sm uppercase tracking-wide text-slate mb-1">Terminal P&amp;L. Mean-variance vs your risk profile</h3>
            <PnlHistogram mv={pnl?.L_mv} asym={pnl?.L_as} tau={pnl?.tau} />
            <div className="flex gap-6 text-xs text-slate mt-1">
              <span><span className="inline-block w-3 h-2 bg-fill border border-grid mr-1" />mean-variance</span>
              <span><span className="inline-block w-3 border-t-2 border-ink mr-1 align-middle" />your profile (downside-averse)</span>
            </div>
          </section>

          <div className="grid sm:grid-cols-2 gap-6">
            <section>
              <h3 className="text-sm uppercase tracking-wide text-slate mb-2">The numbers</h3>
              <Stat label="variance" mv={pnl?.stats_mv.variance} as={pnl?.stats_as.variance} />
              <Stat label="95% CVaR (loss tail)" mv={pnl?.stats_mv.cvar95} as={pnl?.stats_as.cvar95} />
              <Stat label="P[loss > τ]" mv={pnl?.stats_mv.p_exceed} as={pnl?.stats_as.p_exceed} />
              <Stat label="skew" mv={pnl?.stats_mv.skew} as={pnl?.stats_as.skew} />
              <p className="text-xs text-slate mt-2 font-serif">A downside-averse profile accepts more variance to thin the loss tail. Lower CVaR, more negative skew.</p>
            </section>

            <section>
              <h3 className="text-sm uppercase tracking-wide text-slate mb-2">Optimal holdings θ<sub>t</sub></h3>
              <Holdings holdings={hedge?.holdings} />
              <p className="text-xs text-slate mt-1 font-serif">
                {derivative === 'forward'
                  ? 'A forward is replicated by a static unit holding (θ = 1) on every path — there is nothing to react to. Choose the Asian or variance swap to see the holdings move with the path.'
                  : 'A feedback rule: the position reacts to the realised path (three sample paths shown). Hover to read the holding at any time.'}
              </p>
            </section>
          </div>

          <details className="text-xs text-slate border-t border-divider pt-3">
            <summary className="cursor-pointer">Assumptions and notes</summary>
            <p className="font-serif mt-1 max-w-2xl">
              Data is generated from the chosen model (GBM or Heston), under the risk-neutral measure, in
              the stationary regime where the method is validated. In practice, you can parameterize this 
              regime with market factors and adapt the strategy over time.

              
            </p>
          </details>
        </div>
      </div>

      <section className="mt-8 border-t border-ink pt-5">
        <h2 className="text-lg font-semibold mb-1">Portfolio optimization · three correlated assets</h2>
        <p className="font-serif text-slate text-sm max-w-3xl mb-4">
          The same machinery hedges a basket of three correlated assets. The optimal holdings form
          a portfolio whose dollar allocation reallocates over the life of the trade as the assets
          diverge. It is traced below on the simplex: each corner is one asset, a point inside is
          the mix. The bold path is your downside-averse profile; the dashed path is mean-variance.
          The same risk attitude that thins the loss tail also tilts the portfolio.
        </p>
        <div className="grid md:grid-cols-[380px_1fr] gap-6 items-center">
          <Simplex data={portfolio} />
          <div className="text-sm font-serif text-slate space-y-3">
            <p>
              The trade starts at the basket weights (open marker) and ends at the filled marker.
              The line between is the allocation drifting as prices move and the feedback rule
              reallocates toward and away from each asset.
            </p>
            <div className="font-mono text-xs space-y-1">
              <div><span className="inline-block w-4 border-t-2 border-ink mr-2 align-middle" />downside-averse allocation (your profile)</div>
              <div><span className="inline-block w-4 border-t border-slate border-dashed mr-2 align-middle" />mean-variance allocation</div>
              <div><span className="inline-block w-2 h-2 rounded-full border border-ink mr-2 align-middle" />start = basket weights {portfolio ? `(${portfolio.weights.map(w => w.toFixed(2)).join(', ')})` : ''}</div>
            </div>
            <p className="text-xs">
              Two fainter paths show other market scenarios. Hover the bold path to read the mix at
              any time. The example is a fixed 3-asset world (correlated GBM, coarse rebalancing);
              it responds to the risk profile, the one control that is the point.
            </p>
          </div>
        </div>
      </section>
     </div>
    </div>
  )
}

function Step({ n, title, hint, children }) {
  return (
    <section>
      <div className="flex items-center gap-2 mb-2">
        <span className="font-mono text-xs w-5 h-5 grid place-items-center border border-ink rounded-full">{n}</span>
        <h2 className="font-semibold text-sm">{title}</h2>
      </div>
      <p className="text-xs text-slate font-serif mb-2">{hint}</p>
      <div className="space-y-2">{children}</div>
    </section>
  )
}

function Slider({ label, value, min, max, step, onChange }) {
  return (
    <label className="block">
      <div className="flex justify-between text-xs"><span className="text-slate">{label}</span><span className="font-mono">{Number(value).toFixed(2)}</span></div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        className="w-full accent-ink" />
    </label>
  )
}
