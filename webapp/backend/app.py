"""Thin FastAPI wrapper over sigcore for the hedging demo.

Endpoints return only real sigcore computation. /hedge and /pnl share one cached
computation keyed by the parameter hash (fitting + simulating is not instant).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import service

app = FastAPI(title="Signature Hedging Demo")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class Request(BaseModel):
    model: str = "heston"
    params: dict = Field(default_factory=dict)
    derivative: str = "asian"
    K: float | None = None
    gamma: float = 0.0
    delta: float = 0.0
    depth: int = 1
    rebalance_steps: int = 12
    n_paths: int = 8000


def _key(req: Request) -> str:
    return hashlib.sha1(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()


_CACHE: dict[str, dict] = {}


def _compute(req: Request) -> dict:
    k = _key(req)
    if k not in _CACHE:
        if len(_CACHE) > 256:
            _CACHE.clear()
        K, p0, se = service.price_and_strike(req.model, req.params, req.derivative, req.K)
        r = service.hedge_and_pnl(
            req.model, req.params, req.derivative, K, p0, req.gamma, req.delta,
            depth=req.depth, rebalance_steps=req.rebalance_steps, n_paths=req.n_paths)
        r["K"], r["p0"], r["p0_se"] = K, p0, se
        _CACHE[k] = r
    return _CACHE[k]


@app.get("/derivatives")
def derivatives():
    return {k: {"label": v["label"], "blurb": v["blurb"],
                "covector": v["covector"], "level": v["level"],
                "complete": v["complete"]}
            for k, v in service.DERIVATIVES.items()}


@app.post("/paths")
def paths(req: Request):
    return service.sample_paths(req.model, req.params)


@app.post("/price")
def price(req: Request):
    K, p0, se = service.price_and_strike(req.model, req.params, req.derivative, req.K)
    return {"p0": p0, "se": se, "K": K}


@app.post("/hedge")
def hedge(req: Request):
    r = _compute(req)
    return {"p0": r["p0"], "K": r["K"], "ell_mv": r["ell_mv"],
            "ell_as": r["ell_as"], "holdings": r["holdings"], "complete": r["complete"]}


@app.post("/pnl")
def pnl(req: Request):
    r = _compute(req)
    return {"p0": r["p0"], "K": r["K"], "p0_se": r.get("p0_se"), "tau": r["tau"],
            "L_mv": r["L_mv"], "L_as": r["L_as"], "L_unhedged": r["L_unhedged"],
            "stats_mv": r["stats_mv"], "stats_as": r["stats_as"],
            "stats_unhedged": r["stats_unhedged"], "complete": r["complete"]}


_PORT_CACHE: dict[str, dict] = {}


@app.post("/portfolio")
def portfolio(req: Request):
    # depends only on the risk profile (the 3-asset example is fixed)
    key = f"{round(req.gamma, 4)}:{round(req.delta, 4)}"
    if key not in _PORT_CACHE:
        if len(_PORT_CACHE) > 128:
            _PORT_CACHE.clear()
        _PORT_CACHE[key] = service.portfolio(req.gamma, req.delta)
    return _PORT_CACHE[key]


@app.get("/health")
def health():
    return {"ok": True}
