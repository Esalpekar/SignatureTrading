# Signature Trading

Hedging a portfolio of arbitrary, path-dependent derivatives under a risk
profile chosen by the user. A Python core plus an interactive
web demo, built on the signature methods of Lyons et al. (2019).

**Live demo:** https://esalpekar.github.io/SignatureTrading/

---

## The problem with exotic derivatives

Many derivatives have simple rules. A call option pays X_T − K, the asset's price at maturity minus a fixed strike. But as a contract's rules get more exotic, the mathematics needed to analyze it tends to grow quickly, and each new payoff often seems to need its own bespoke treatment. The effect is that the tractable contracts end up well understood while the interesting ones do not.

This project hedges derivatives whose payoff is a continuous function of the asset's price path X_t, rather than only its final value. A call option reads a single number off the path, its endpoint. A variance swap reads how much the price moved over the window, regardless of where it ended up. The two contracts have little in common, yet both are continuous functions of the same path, and that is the only property the method needs.

What's more, the underlying theory conveniently lets you optimize for an
arbitrary attitude toward risk. Much of the portfolio-hedging literature
descends from mean-variance hedging, pioneered more than sixty years ago (
it shows its age...) Mean-variance penalizes upside and downside symmetrically: a
surprise gain is treated as exactly as regrettable as an equal surprise loss.
Real traders are not so even-handed.

Here, a trader who fears losses more than they enjoy gains, or who has some
situational need to shape a particular higher-order feature of the
return distribution, can hedge directly under that preference. The demo takes this preference as an input and shows the resulting
hedge side by side with the mean-variance answer, so you can watch the
difference.

## Signature theory

The workhorse is signature theory. A signature is a representation of a path in the form of many coefficients, arranged in levels.

Take the asset's price path and enrich it slightly: add a clock, and a
"lead-lag" copy of the path against itself (a standard trick that makes
volatility visible to the integrals below). The levels are summarized as follows:

- **Level 1** records the net change in each channel (where the path ended up).
- **Level 2** records signed areas (how pairs of assets happen to co-move over time).
  This captures order and interaction that a net change throws away (it is, for
  instance, how realized variance becomes legible).
- **Level 3** records a still finer interaction, and so on up.

Truncate at some level (they get proressively less influential, so this is usually defensible) and you have a finite feature vector that summarizes the entire path.

There's a very convenient property that makes all of this useful: any
continuous function of the path can be written, to arbitrary accuracy, as a
plain linear combination of signature terms. A nonlinear, history-dependent
quantity can be written as one linear function. The signature linearizes
the space of everything you might want to compute from a path.

## Why this works in hedging

Two consequences follow:

First, a trading strategy is itself one of those functions. At each time
`t < T`, how much of an asset you hold can depend on the entire path so far.
So a strategy is just a linear readout of
the running signature `S_{0,t}`: pick a list of coefficients `ℓ`, and your
holding at time `t` is the inner product `⟨ℓ, S_{0,t}⟩`. The same coefficients,
evaluated as the path unfolds, generate the whole time series of holdings.

Secondly, the product of two
signature-linear functions is again signature-linear (via an operation called
the shuffle product). Follow the chain: your loss `L` at maturity is linear in
the signature; your risk penalty `P(L)` (the polynomial encoding how much you
dislike a loss of each size) is a polynomial in `L`, and so, after shuffling,
still linear in the signature. Its expectation therefore collapses to a single
inner product:

```
E[ P(L) ]  =  ⟨ P^shuffle(ℓ),  E[S] ⟩
```

where `E[S]` is the expected signature of the asset's price path. You get `E[S]` from whatever describes the price dynamics:
GBM, stochastic-volatility Heston, or anything you can simulate (closed-form for
some models, Monte Carlo otherwise). Everything downstream is model-agnostic
algebra.

So the problem "minimize my expected risk penalty over all possible feedback
strategies" becomes "minimize one inner product over
the coefficients `ℓ`, against a fixed vector `E[S]`." This is a finite, well-posed
optimization. That is the trick the whole project rests on.

## The demo

The [live demo](https://esalpekar.github.io/SignatureTrading/) lets you turn the
knobs:

- pick a **model of the underlying process** — GBM or stochastic-volatility Heston;
- pick a **contract** — forward, Asian (average-price), or variance swap;
- dial your **risk profile** — the loss polynomial, i.e. how steeply you punish
  particular kinds of losses.

It then derives the optimal hedge, plots the resulting distribution of profit
and loss, and overlays what plain mean-variance hedging would have done instead.
Drag the risk sliders and watch the loss tail retreat. (It also traces the
optimal allocation of a three-asset basket on a simplex, for the multi-asset
case.)

## Repository

- `sigcore/` — the library: signatures, models (GBM, Heston), pricing, hedging.
- `tests/` — acceptance and unit tests pinning the core to analytic ground truth.
- `report.py` — validation harness (closed-form vs. Monte Carlo checks).
- `webapp/` — React frontend + FastAPI backend. The live demo runs off a
  precomputed static databank (`webapp/build_databank.py`), so it needs no server.
- `background/` — the source papers behind the method.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # tests
python report.py   # validation report
./webapp/run.sh    # local web app (frontend + backend)
```

## References

- Lyons, Nejad & Pérez Arribas, *Nonparametric pricing and hedging of exotic
  derivatives* (2019).
- *Signature Trading: A Path-Dependent Extension of the Mean–Variance Framework
  with Exogenous Signals (2024).*

Both are in [`background/`](background/).
