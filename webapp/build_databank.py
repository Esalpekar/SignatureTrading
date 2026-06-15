"""Precompute the entire (small) input space of the demo into a static databank.

The live app only varies four things — model, derivative, gamma, delta — and the
fair price/strike depend on just (model, derivative). So we enumerate the grid,
run the real sigcore computation for every point, and write compact JSON the
static frontend reads instead of calling the FastAPI backend.

Histograms are pre-binned to 161 bins (exactly matching charts.jsx) so each
config payload is ~2 KB instead of ~400 KB of raw loss samples.

Run from the project venv:
    PYTHONPATH=webapp .venv/bin/python webapp/build_databank.py
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from backend import service

# --- fixed UI parameters (mirror frontend/src/App.jsx) -----------------------
HESTON = {"x0": 1.0, "v0": 0.04, "r": 0.0, "kappa": 2.0, "theta": 0.04,
          "xi": 0.6, "rho": -0.8, "T": 1.0}
GBM = {"x0": 1.0, "r": 0.0, "sigma": 0.2, "T": 1.0}
MODELS = {"heston": {"params": HESTON, "rebalance_steps": 3},
          "gbm": {"params": GBM, "rebalance_steps": 4}}
DERIVS = ["forward", "asian", "varswap"]
DEPTH = 1
N_PATHS = 6000
BINS = 161  # must match PnlHistogram in charts.jsx

# --- grid (~50 ticks per slider; the sliders snap to these) ------------------
GAMMA = [round(0 + i * 1.0, 3) for i in range(31)]      # 0 .. 30 step 1
DELTA = [round(0 + i * 10.0, 3) for i in range(41)]     # 0 .. 400 step 10


def delta_floor(g: float) -> float:
    return 3.0 * g * g / 8.0


def valid_deltas(g: float):
    """Indices/values of the delta grid above the convexity floor for this gamma."""
    floor = delta_floor(g)
    return [(j, d) for j, d in enumerate(DELTA) if d >= floor]


def r(x, n=5):
    return round(float(x), n)


def prebin(l_mv, l_as):
    """Replicate charts.jsx histPair: combined [lo,hi], 161 bins, clamped."""
    a = np.asarray(l_mv)
    b = np.asarray(l_as)
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi - lo < 1e-9:
        hi = lo + 1e-6
    w = (hi - lo) / BINS

    def counts(arr):
        idx = np.floor((arr - lo) / w).astype(int)
        np.clip(idx, 0, BINS - 1, out=idx)
        return np.bincount(idx, minlength=BINS)[:BINS].tolist()

    return {"lo": r(lo, 6), "hi": r(hi, 6), "mv": counts(a), "asym": counts(b)}


# --- workers (top-level for ProcessPoolExecutor pickling) --------------------

def _hedge_entry(task):
    model, deriv, K, p0, gi, gamma, di, delta = task
    cfg = MODELS[model]
    res = service.hedge_and_pnl(
        model, cfg["params"], deriv, K, p0, gamma, delta,
        depth=DEPTH, rebalance_steps=cfg["rebalance_steps"], n_paths=N_PATHS)
    hist = prebin(res["L_mv"], res["L_as"])

    def stat(s):
        return {"variance": r(s["variance"], 6), "cvar95": r(s["cvar95"], 6),
                "p_exceed": r(s["p_exceed"], 6), "skew": r(s["skew"], 6)}

    entry = {
        "tau": r(res["tau"], 6),
        "hist": hist,
        "stats_mv": stat(res["stats_mv"]),
        "stats_as": stat(res["stats_as"]),
        "theta": [[r(v, 5) for v in h["theta"]] for h in res["holdings"]],
    }
    # t is identical across all configs of a model; capture once
    t = [r(v, 5) for v in res["holdings"][0]["t"]]
    return model, deriv, gi, di, entry, t, bool(res["complete"])


def _portfolio_entry(task):
    gi, gamma, di, delta = task
    p = service.portfolio(gamma, delta)

    def trajs(key):
        return [{"t": [r(v, 4) for v in tr["t"]],
                 "w": [[r(x, 4) for x in node] for node in tr["w"]]}
                for tr in p[key]]

    return gi, di, {"mv": trajs("mv"), "asym": trajs("asym")}


def main():
    out = os.path.join(os.path.dirname(__file__), "frontend", "public", "data")
    os.makedirs(out, exist_ok=True)

    # 1. derivatives metadata (static)
    derivs_meta = {k: {"label": v["label"], "blurb": v["blurb"],
                       "covector": v["covector"], "level": v["level"],
                       "complete": v["complete"]}
                   for k, v in service.DERIVATIVES.items()}
    with open(os.path.join(out, "derivatives.json"), "w") as f:
        json.dump(derivs_meta, f, separators=(",", ":"))

    # 2. sample price paths per model
    for model, cfg in MODELS.items():
        paths = service.sample_paths(model, cfg["params"])
        paths = {"t": [r(v, 5) for v in paths["t"]],
                 "paths": [[r(v, 5) for v in p] for p in paths["paths"]],
                 "terminal": [r(v, 5) for v in paths["terminal"]]}
        with open(os.path.join(out, f"paths_{model}.json"), "w") as f:
            json.dump(paths, f, separators=(",", ":"))

    # 3. price/strike per (model, deriv) + grid metadata
    meta = {"gamma": GAMMA, "delta": DELTA, "depth": DEPTH, "n_paths": N_PATHS,
            "prices": {}}
    hedge_tasks = []
    for model, cfg in MODELS.items():
        for deriv in DERIVS:
            K, p0, se = service.price_and_strike(model, cfg["params"], deriv, None)
            meta["prices"][f"{model}_{deriv}"] = {"K": r(K, 6), "p0": r(p0, 6),
                                                  "se": r(se, 6)}
            for gi, g in enumerate(GAMMA):
                for di, d in valid_deltas(g):
                    hedge_tasks.append((model, deriv, K, p0, gi, g, di, d))

    # 4. run all hedge/pnl points in parallel, assemble per-(model,deriv) files
    files = {f"{m}_{d}": {"entries": {}} for m in MODELS for d in DERIVS}
    print(f"computing {len(hedge_tasks)} hedge/pnl points ...", flush=True)
    done = 0
    with ProcessPoolExecutor() as ex:
        for model, deriv, gi, di, entry, t, complete in ex.map(_hedge_entry, hedge_tasks, chunksize=8):
            key = f"{model}_{deriv}"
            files[key]["entries"][f"{gi}_{di}"] = entry
            files[key]["t"] = t
            files[key]["complete"] = complete
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(hedge_tasks)}", flush=True)

    for key, payload in files.items():
        model, deriv = key.split("_", 1)
        payload["K"] = meta["prices"][key]["K"]
        payload["p0"] = meta["prices"][key]["p0"]
        with open(os.path.join(out, f"{key}.json"), "w") as f:
            json.dump(payload, f, separators=(",", ":"))

    # 5. portfolio depends only on (gamma, delta) — one shared file
    seen = {}
    port_tasks = []
    for gi, g in enumerate(GAMMA):
        for di, d in valid_deltas(g):
            if (gi, di) not in seen:
                seen[(gi, di)] = True
                port_tasks.append((gi, g, di, d))
    print(f"computing {len(port_tasks)} portfolio points ...", flush=True)
    sample = service.portfolio(GAMMA[0], next(d for _, d in valid_deltas(GAMMA[0])))
    port = {"labels": sample["labels"], "weights": [r(w, 4) for w in sample["weights"]],
            "entries": {}}
    done = 0
    with ProcessPoolExecutor() as ex:
        for gi, di, entry in ex.map(_portfolio_entry, port_tasks, chunksize=8):
            port["entries"][f"{gi}_{di}"] = entry
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(port_tasks)}", flush=True)
    with open(os.path.join(out, "portfolio.json"), "w") as f:
        json.dump(port, f, separators=(",", ":"))

    # 6. manifest the frontend reads first
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))

    # report sizes
    total = 0
    for name in sorted(os.listdir(out)):
        sz = os.path.getsize(os.path.join(out, name))
        total += sz
        print(f"  {name:28s} {sz/1024:8.1f} KB")
    print(f"databank total: {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
