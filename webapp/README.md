# Signature Hedging — Web Demo

A monochrome React + FastAPI demo over the validated `sigcore` library. Pick a
**world** (GBM / Heston), a **contract** (forward, Asian, variance swap), and a
**risk attitude** (the loss polynomial), and watch the optimal hedge and how the
risk attitude reshapes the loss tail. **Every number is a live `sigcore`
computation** — no mocked data.

The hero interaction — dragging the risk-profile sliders and watching the
terminal-P&L loss tail retreat — is only alive in an **incomplete market**, so
the demo defaults to **Heston** (or coarse-rebalanced GBM). In a complete market
(fine-grid GBM) the risk profile does nothing; that is the mathematics (test
gates H6/MH6), and the UI surfaces a note rather than letting it look broken.

## Run

The demo is **two servers**: a FastAPI backend on `:8000` and the Vite dev server
(which proxies `/api` → `:8000`). If you only start the frontend you'll see
`ECONNREFUSED 127.0.0.1:8000` — that just means the backend isn't up.

One command (starts both, stops the backend on exit):

```bash
./webapp/run.sh
```

Or manually. **The backend must run from the project `.venv`** — that is where
`sigcore`, `iisignature`, and `fastapi` are installed (a bare `pip install` may
land in system Python, which has no `sigcore`):

```bash
source .venv/bin/activate
pip install -r webapp/requirements.txt        # fastapi, uvicorn (sigcore already there)
PYTHONPATH=webapp python -m uvicorn backend.app:app --reload --port 8000   # terminal 1
cd webapp/frontend && npm install && npm run dev                            # terminal 2
```

`npm run build` produces a static bundle in `dist/`.

## Endpoints (thin wrappers over `sigcore`)

- `GET  /derivatives` — preset contracts, each with its closed-form covector.
- `POST /paths` — sample price paths + terminal distribution for the model visual.
- `POST /price` — fair price `p₀` (= `e^{-rT}·E[payoff]` by Monte Carlo).
- `POST /hedge` — fitted strategy `ℓ*` (mean-variance and selected polynomial) + holdings `θ_t` on sample paths.
- `POST /pnl` — shortfall distributions for both hedges + summary numbers (variance, 95% CVaR, `P[loss>τ]`, skew).

## Notes on the math (carried from the library)

- The hedge is fit by the per-path objective `E[P(L)]` — equal to the shuffle
  objective `⟨P^⧢(𝓛), E[S]⟩` by the shuffle identity — using the running-signature
  feedback `θ_t = ⟨ℓ, S_{0,t}⟩` validated in the hedging phase.
- The tail-reshaping is clearest at **strategy depth 0** (a static position has
  room to trade variance for tail); at higher depth the hedge approaches
  replication and the reshaping shrinks — the honest depth/completeness tradeoff,
  exposed as a control.
- Convexity (`δ ≥ 3γ²/8`) is enforced in the sliders, keeping the penalty inside
  the validated regime.
