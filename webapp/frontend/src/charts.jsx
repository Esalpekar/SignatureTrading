// Hand-rolled monochrome SVG charts. Series are distinguished by weight/texture
// and dash. Never colour (the grayscale brief makes that load-bearing).
import React, { useRef, useState } from 'react'

const INK = '#141414', GRID = '#B8B8B8', SLATE = '#6B6B6B', FILL = '#ECECEC'

function counts(values, lo, hi, bins) {
  const c = new Array(bins).fill(0)
  const w = (hi - lo) / bins
  for (const v of values) {
    let b = Math.floor((v - lo) / w)
    if (b < 0) b = 0
    if (b >= bins) b = bins - 1
    c[b]++
  }
  return { c, w }
}

// Both distributions share ONE vertical scale (same n_paths), so the thinned
// loss tail and widened body are read off honestly rather than each re-normalised.
function histPair(mv, asym, lo, hi, bins) {
  const m = counts(mv, lo, hi, bins), a = counts(asym, lo, hi, bins)
  const max = Math.max(...m.c, ...a.c, 1)
  const bar = (cc, i) => ({ x0: lo + i * m.w, x1: lo + (i + 1) * m.w, h: cc / max, c: cc })
  return { mv: m.c.map(bar), asym: a.c.map(bar), w: m.w }
}

// map a pointer event to a viewBox x-coordinate (the svg scales responsively)
function vbX(e, ref, width) {
  const r = ref.current.getBoundingClientRect()
  return ((e.clientX - r.left) / r.width) * width
}

// Overlaid terminal-P&L distributions: MV (solid light fill) vs your risk
// profile (heavy outline). The loss tail (L > tau) is shaded. THIS is the hero.
export function PnlHistogram({ mv, asym, tau, width = 620, height = 300 }) {
  const ref = useRef(null)
  const [hx, setHx] = useState(null)
  if (!mv || !asym) return null
  const all = mv.concat(asym)
  const lo = Math.min(...all), hi = Math.max(...all)
  const bins = 161
  const hist = histPair(mv, asym, lo, hi, bins)
  const hm = { bars: hist.mv, w: hist.w }, ha = { bars: hist.asym, w: hist.w }
  const pad = { l: 8, r: 8, t: 14, b: 34 }
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b
  const sx = x => pad.l + ((x - lo) / (hi - lo)) * W
  const sy = h => pad.t + (1 - h) * H
  const bw = W / bins
  const taux = sx(tau)

  const onMove = e => { const x = vbX(e, ref, width); if (x >= pad.l && x <= pad.l + W) setHx(x) }
  let tip = null
  if (hx != null) {
    const L = lo + ((hx - pad.l) / W) * (hi - lo)
    const bi = Math.min(bins - 1, Math.max(0, Math.floor((L - lo) / hm.w)))
    tip = { L, mv: hm.bars[bi].c, as: ha.bars[bi].c }
  }
  return (
    <svg ref={ref} viewBox={`0 0 ${width} ${height}`} className="w-full"
      onMouseMove={onMove} onMouseLeave={() => setHx(null)} role="img"
      aria-label="Terminal profit-and-loss distribution: mean-variance versus selected risk profile">
      <rect x={taux} y={pad.t} width={Math.max(0, pad.l + W - taux)} height={H} fill={FILL} opacity="0.6" />
      <line x1={taux} y1={pad.t} x2={taux} y2={pad.t + H} stroke={SLATE} strokeDasharray="3 3" />
      <text x={taux + 4} y={pad.t + 10} fontSize="9" fill={SLATE} className="font-mono">loss &gt; τ</text>
      {hm.bars.map((b, i) => (
        <rect key={'m' + i} x={sx(b.x0)} y={sy(b.h)} width={bw - 0.3} height={H - (sy(b.h) - pad.t)}
          fill={FILL} stroke={GRID} strokeWidth="0.3" />
      ))}
      <polyline fill="none" stroke={INK} strokeWidth="1.6"
        points={ha.bars.map(b => `${sx(b.x0)},${sy(b.h)} ${sx(b.x1)},${sy(b.h)}`).join(' ')} />
      <line x1={pad.l} y1={pad.t + H} x2={pad.l + W} y2={pad.t + H} stroke={GRID} />
      <text x={pad.l} y={height - 8} fontSize="9" fill={SLATE} className="font-mono">← gain</text>
      <text x={pad.l + W} y={height - 8} fontSize="9" fill={SLATE} textAnchor="end" className="font-mono">loss (shortfall &gt; 0) →</text>
      {hx != null && (
        <g>
          <line x1={hx} y1={pad.t} x2={hx} y2={pad.t + H} stroke={INK} strokeWidth="0.6" />
          <g transform={`translate(${Math.min(hx + 6, pad.l + W - 96)}, ${pad.t + 4})`}>
            <rect width="96" height="34" fill="#fff" stroke={GRID} />
            <text x="5" y="12" fontSize="9" className="font-mono" fill={INK}>L = {tip.L.toFixed(3)}</text>
            <text x="5" y="23" fontSize="9" className="font-mono" fill={SLATE}>MV n={tip.mv}</text>
            <text x="5" y="32" fontSize="9" className="font-mono" fill={INK}>profile n={tip.as}</text>
          </g>
        </g>
      )}
    </svg>
  )
}

// P(x) = x^2 + gamma x^3 + delta x^4. The y-axis auto-fits the current curve
// (so it never clips off the top), and the x-span is kept narrow so the bowl's
// asymmetry and the leftward shift of its minimum are the dominant, visibly
// moving features as the sliders change. Convexity is preserved by the controls.
export function PolyPlot({ gamma, delta, width = 240, height = 150, span = 1.3 }) {
  const xs = []
  for (let x = -span; x <= span + 1e-4; x += span / 80) xs.push(x)
  const P = x => x * x + gamma * x ** 3 + delta * x ** 4
  const ys = xs.map(P)
  const yMax = Math.max(...ys, 1e-6) * 1.08
  const pad = 6
  const sx = x => pad + ((x + span) / (2 * span)) * (width - 2 * pad)
  const sy = y => height - pad - (y / yMax) * (height - 2 * pad)
  const path = xs.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(' ')
  // mark the minimum so its leftward drift under a downside tilt is legible
  const imin = ys.indexOf(Math.min(...ys))
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Loss penalty shape">
      <line x1={sx(0)} y1={pad} x2={sx(0)} y2={height - pad} stroke={GRID} strokeWidth="0.5" />
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke={GRID} strokeWidth="0.5" />
      <path d={path} fill="none" stroke={INK} strokeWidth="1.8" />
      <circle cx={sx(xs[imin])} cy={sy(ys[imin])} r="2" fill={INK} />
      <text x={width - pad} y={height - pad - 2} fontSize="9" fill={SLATE} textAnchor="end" className="font-mono">loss →</text>
      <text x={pad} y={height - pad - 2} fontSize="9" fill={SLATE} className="font-mono">← gain</text>
    </svg>
  )
}

// theta_t for a few sample paths. A feedback rule, not a fixed schedule.
export function Holdings({ holdings, width = 300, height = 170 }) {
  const ref = useRef(null)
  const [hi_, setHi] = useState(null)
  if (!holdings || !holdings.length) return null
  const all = holdings.flatMap(h => h.theta)
  const lo = Math.min(...all), hi = Math.max(...all, lo + 1e-6)
  const t = holdings[0].t
  const pad = { l: 6, r: 6, t: 10, b: 22 }
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b
  const sx = i => pad.l + (i / (t.length - 1)) * W
  const sy = v => pad.t + (1 - (v - lo) / (hi - lo)) * H
  const dashes = ['', '4 3', '1 3']
  const onMove = e => {
    const x = vbX(e, ref, width)
    const i = Math.round(((x - pad.l) / W) * (t.length - 1))
    setHi(Math.min(t.length - 1, Math.max(0, i)))
  }
  return (
    <svg ref={ref} viewBox={`0 0 ${width} ${height}`} className="w-full"
      onMouseMove={onMove} onMouseLeave={() => setHi(null)} role="img" aria-label="Optimal holdings over time">
      <line x1={pad.l} y1={sy(0)} x2={pad.l + W} y2={sy(0)} stroke={GRID} strokeWidth="0.5" />
      {holdings.map((h, k) => (
        <polyline key={k} fill="none" stroke={INK} strokeWidth="1.2" strokeDasharray={dashes[k % 3]}
          points={h.theta.map((v, i) => `${sx(i)},${sy(v)}`).join(' ')} />
      ))}
      <text x={pad.l} y={height - 6} fontSize="9" fill={SLATE} className="font-mono">t = 0</text>
      <text x={pad.l + W} y={height - 6} fontSize="9" fill={SLATE} textAnchor="end" className="font-mono">T</text>
      {hi_ != null && (
        <g>
          <line x1={sx(hi_)} y1={pad.t} x2={sx(hi_)} y2={pad.t + H} stroke={INK} strokeWidth="0.6" />
          {holdings.map((h, k) => (
            <circle key={k} cx={sx(hi_)} cy={sy(h.theta[hi_])} r="2.2" fill={INK} />
          ))}
          <g transform={`translate(${Math.min(sx(hi_) + 6, pad.l + W - 90)}, ${pad.t + 2})`}>
            <rect width="90" height={12 + 10 * holdings.length} fill="#fff" stroke={GRID} />
            <text x="5" y="10" fontSize="9" className="font-mono" fill={SLATE}>t={t[hi_].toFixed(2)}</text>
            {holdings.map((h, k) => (
              <text key={k} x="5" y={20 + 10 * k} fontSize="9" className="font-mono" fill={INK}>
                θ{k + 1} = {h.theta[hi_].toFixed(3)}
              </text>
            ))}
          </g>
        </g>
      )}
    </svg>
  )
}

// The 2-simplex: each vertex is an asset, an interior point is a portfolio
// allocation (w_A, w_B, w_C summing to 1). The optimal hedge's dollar allocation
// is traced over the life of the trade; the downside-averse path (bold) is shown
// against the mean-variance path (dashed) so the risk profile's tilt is visible.
export function Simplex({ data, width = 380, height = 360 }) {
  const ref = useRef(null)
  const [hover, setHover] = useState(null)
  if (!data) return null
  const { labels, mv, asym, weights } = data
  const pad = 38
  const W = width - 2 * pad, H = height - 2 * pad
  const V = [[pad + W / 2, pad], [pad, pad + H], [pad + W, pad + H]]   // A top, B left, C right
  const bary = w => [
    w[0] * V[0][0] + w[1] * V[1][0] + w[2] * V[2][0],
    w[0] * V[0][1] + w[1] * V[1][1] + w[2] * V[2][1],
  ]
  const poly = traj => traj.w.map(w => bary(w).join(',')).join(' ')
  const start = bary(weights)
  const main = asym[0], mainMv = mv[0]
  const endA = bary(main.w[main.w.length - 1])
  const onMove = e => {
    const r = ref.current.getBoundingClientRect()
    const mx = ((e.clientX - r.left) / r.width) * width
    const my = ((e.clientY - r.top) / r.height) * height
    let best = 0, bd = 1e9
    main.w.forEach((w, i) => { const [x, y] = bary(w); const dd = (x - mx) ** 2 + (y - my) ** 2; if (dd < bd) { bd = dd; best = i } })
    setHover(best)
  }
  return (
    <svg ref={ref} viewBox={`0 0 ${width} ${height}`} className="w-full"
      onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
      aria-label="Portfolio allocation trajectory on the asset simplex">
      {[0.25, 0.5, 0.75].map((f, i) => (
        <g key={i} stroke={GRID} strokeWidth="0.4" opacity="0.6">
          <line x1={V[1][0] + f * (V[0][0] - V[1][0])} y1={V[1][1] + f * (V[0][1] - V[1][1])}
            x2={V[2][0] + f * (V[0][0] - V[2][0])} y2={V[2][1] + f * (V[0][1] - V[2][1])} />
          <line x1={V[0][0] + f * (V[1][0] - V[0][0])} y1={V[0][1] + f * (V[1][1] - V[0][1])}
            x2={V[2][0] + f * (V[1][0] - V[2][0])} y2={V[2][1] + f * (V[1][1] - V[2][1])} />
          <line x1={V[0][0] + f * (V[2][0] - V[0][0])} y1={V[0][1] + f * (V[2][1] - V[0][1])}
            x2={V[1][0] + f * (V[2][0] - V[1][0])} y2={V[1][1] + f * (V[2][1] - V[1][1])} />
        </g>
      ))}
      <polygon points={V.map(v => v.join(',')).join(' ')} fill="none" stroke={INK} strokeWidth="1" />
      {asym.slice(1).map((tr, k) => (
        <polyline key={'s' + k} fill="none" stroke={SLATE} strokeWidth="0.8" opacity="0.55"
          strokeDasharray={k ? '1 3' : ''} points={poly(tr)} />
      ))}
      <polyline fill="none" stroke={SLATE} strokeWidth="1" strokeDasharray="4 3" points={poly(mainMv)} />
      <polyline fill="none" stroke={INK} strokeWidth="2" points={poly(main)} />
      <circle cx={start[0]} cy={start[1]} r="3.2" fill="#fff" stroke={INK} strokeWidth="1.2" />
      <circle cx={endA[0]} cy={endA[1]} r="3.2" fill={INK} />
      <text x={V[0][0]} y={V[0][1] - 8} fontSize="11" textAnchor="middle" className="font-mono" fill={INK}>{labels[0]}</text>
      <text x={V[1][0] - 4} y={V[1][1] + 14} fontSize="11" textAnchor="middle" className="font-mono" fill={INK}>{labels[1]}</text>
      <text x={V[2][0] + 4} y={V[2][1] + 14} fontSize="11" textAnchor="middle" className="font-mono" fill={INK}>{labels[2]}</text>
      {hover != null && (() => {
        const [hxp, hyp] = bary(main.w[hover]); const w = main.w[hover]; const th = main.t[hover]
        return (
          <g>
            <circle cx={hxp} cy={hyp} r="3" fill="#fff" stroke={INK} strokeWidth="1.4" />
            <g transform={`translate(${Math.min(hxp + 6, width - 96)}, ${Math.max(hyp - 36, 2)})`}>
              <rect width="92" height="44" fill="#fff" stroke={GRID} />
              <text x="5" y="11" fontSize="9" className="font-mono" fill={SLATE}>t = {th.toFixed(2)}</text>
              <text x="5" y="22" fontSize="9" className="font-mono" fill={INK}>{labels[0]} {(w[0] * 100).toFixed(0)}%</text>
              <text x="5" y="32" fontSize="9" className="font-mono" fill={INK}>{labels[1]} {(w[1] * 100).toFixed(0)}%</text>
              <text x="5" y="42" fontSize="9" className="font-mono" fill={INK}>{labels[2]} {(w[2] * 100).toFixed(0)}%</text>
            </g>
          </g>
        )
      })()}
    </svg>
  )
}

export function PricePaths({ data, width = 300, height = 160 }) {
  if (!data) return null
  const { t, paths } = data
  const all = paths.flat()
  const lo = Math.min(...all), hi = Math.max(...all)
  const pad = { l: 6, r: 6, t: 10, b: 16 }
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b
  const sx = i => pad.l + (i / (t.length - 1)) * W
  const sy = v => pad.t + (1 - (v - lo) / (hi - lo)) * H
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Sample price paths">
      {paths.map((p, k) => (
        <polyline key={k} fill="none" stroke={SLATE} strokeWidth="0.7" opacity="0.7"
          points={p.map((v, i) => `${sx(i)},${sy(v)}`).join(' ')} />
      ))}
    </svg>
  )
}
