# Signature Pricing Core — Implementation Spec (v0: Validation Harness)

> Audience: a coding agent that will implement this end to end.
> Mandate: **build only what the math below requires.** Anything not needed to (a) compute an expected signature two independent ways and (b) price a forward and check it against closed form is a non-goal (see §6). Resist adding it.

---

## 0. Purpose, and why this exists before anything else

This harness has exactly one job: **demonstrate that the chain `simulate → embed → expected signature → price` reproduces a known closed-form price on generated data.**

The motivation is not ceremony. Every later capability in this project — hedging, personalised risk profiles, multi-asset portfolios — reads the *same* expected signature and prices or optimises against it. If the path embedding or the expected signature is computed incorrectly, every downstream result will be wrong **and will still look plausible** (a hedge curve, a P&L histogram — nothing visibly breaks). The only defence is to pin the foundation to analytic ground truth before building on it. This harness is that pin.

When all acceptance tests in §3 pass, the core is trusted and the next phase (hedging) may begin.

**Stack.** Python ≥ 3.11; `numpy`; `iisignature` (signature computation); `pytest`. No other runtime dependencies. No plotting, no web framework, no CLI framework. Console output only.

**Do not hand-roll the signature.** Computing iterated integrals via Chen's relation is a solved, bug-prone problem; use `iisignature`. (`esig` is an acceptable fallback. Do **not** use `signatory` — its torch-version coupling is a maintenance trap.)

---

## 1. Mathematical objects (implement exactly these, nothing more)

**Geometric Brownian motion (the data-generating process).** Under the pricing (risk-neutral) measure the drift is the risk-free rate `r`:
`X_{t+Δt} = X_t · exp((r − σ²/2)·Δt + σ·√Δt·Z)`, `Z ~ N(0,1)` i.i.d.
This is the *exact* solution of GBM — use it. Do **not** use Euler–Maruyama; GBM needs no discretised SDE and an Euler scheme only injects avoidable bias.

**Time augmentation.** A scalar price series `X_0..X_n` on grid `t_k = k·Δt` becomes a 2-channel path `(t_k, X_k)`. Channel 0 = time, channel 1 = price. Time augmentation is required so the signature determines the time-parametrised path.

**Truncated signature.** For a discrete `d`-channel path and truncation level `N`, the signature is the vector of iterated integrals of the piecewise-linear interpolation, levels `1..N`. (`iisignature.sig` returns levels `1..N` flattened; **level 0, the empty word ≡ 1, is omitted** — see §7.)

**Expected signature, two ways.**
- *Monte Carlo:* the coordinate-wise mean of the per-path signatures. Error decays like `1/√(n_paths)`; it is **not** reduced by a finer time grid.
- *Analytic (Gaussian case):* for a time-augmented Brownian motion with drift, the expected signature is the **truncated tensor exponential of a generator** `ξ`:
  `E[S_{0,T}] = exp_⊗(T·ξ)`, with `ξ = (drift vector at level 1) + ½·(diffusion matrix at level 2)`.
  For the *driftless unit Brownian* driver `(t, W_t)`: `ξ = e_0 + ½·(e_1 ⊗ e_1)` (time drifts at rate 1 → `e_0`; `W` is driftless → no level-1 term; `W` has unit diffusion → `½ e_1⊗e_1`). This is Fawcett's formula.

**Forward covector.** A forward struck at `K` pays `X_T − K`. As a covector over channels `(time, price)`:
`f = (X_0 − K)·∅ + 1·(price-letter)`, where `∅` is the empty word and the price-letter is the single-letter word selecting channel 1. Then `⟨f, S⟩ = (X_0 − K) + (X_T − X_0) = X_T − K`. ✔

**Price.** `price = e^{−rT}·⟨f, E^Q[S]⟩`. For the forward this needs **only the level-1 price coordinate** of `E^Q[S]`, which in closed form is `E^Q[X_T − X_0] = X_0·(e^{rT} − 1)`. The full GBM expected signature is **not required and must not be built** (it is messy and unused here).

---

## 2. Modules and behaviour

Behaviour, not internals. Signatures are guidance; keep them but optimise freely inside.

### `gbm.py`
- `simulate(x0, r, sigma, T, n_steps, n_paths, seed) -> (times, paths)`
  - Returns `times` shape `(n_steps+1,)` and `paths` shape `(n_paths, n_steps+1)`, using the **exact** GBM recursion above with drift `r`.
  - Deterministic given `seed`.

### `embed.py`
- `time_augment(times, series) -> path` — stack to shape `(..., n_steps+1, 2)`, channel 0 = time, channel 1 = value.
- `lead_lag(series) -> ll` — the **Hoff lead-lag** transform of a scalar series: a 2-channel (lead, lag) path where lead is advanced one sample ahead of lag, traced as a staircase. Construction: see Chevyrev & Kormilitzin, *A Primer on the Signature Method* (or Flint–Hambly–Lyons). **You do not need to memorise the exact node sequence — correctness is defined by test T2**: the antisymmetric part of the level-2 signature of `(lead, lag)` must equal `½ · Σ_k (x_{k+1} − x_k)²` (half the realised quadratic variation).
  - **Note for the agent:** `lead_lag` is **not exercised by the forward-pricing path** (a forward is a linear claim and needs no Itô integral). It is implemented and validated here only because it is the embedding the *hedging* phase will require, and validating it now against the quadratic-variation identity de-risks that phase. **Do not wire it into the pricing path.** How time and lead-lag combine into the hedging embedding is deferred to the next phase.

### `signature.py`
- `sig(path, level) -> np.ndarray` — thin wrapper over `iisignature.sig`; returns the flat level-`1..N` signature.
- `tensor_exp(generator, n_channels, level) -> np.ndarray` — truncated tensor exponential `Σ_{k=0}^{level} A^{⊗k}/k!`, where `A^{⊗k}` is computed by truncated tensor multiplication (drop anything above `level`). The `generator` is supplied as its level-1 and level-2 components. **Output must be flattened in the same word order as `iisignature.sig`** so it is coordinate-aligned for comparison (see §7).
- `word_index(word, n_channels) -> int` — map a word (e.g. the price-letter `(1,)`) to its position in the flat signature vector. Keep this explicit rather than hard-coding offsets.

### `expected_signature.py`
- `monte_carlo(paths_embedded, level) -> np.ndarray` — mean of `sig(path, level)` over paths. Also return the per-coordinate standard error (`std/√n`) for the report.
- `analytic_gaussian(generator, n_channels, level) -> np.ndarray` — `tensor_exp(T·ξ, …)`. Used for the Brownian driver in T1.

### `price.py`
- `forward_covector(x0, K, n_channels) -> (const, vec)` — returns the empty-word coefficient `const = x0 − K` and the body covector `vec` (zeros except `1` at the price-letter index). Kept separate because `iisignature` omits the empty word (§7).
- `price(const, vec, expected_sig, r, T) -> float` — returns `e^{−rT}·(const·1 + vec · expected_sig)`.

---

## 3. Acceptance tests — definition of done

Each test states **what** it checks, **why**, the **analytic target**, and a **tolerance**. The harness is "done" when all pass. Use a fixed seed; report the actual numbers (§4).

**T1 — Signature engine vs Fawcett (Brownian driver).**
Why: confirms `sig` + Monte-Carlo averaging are correct against the one process whose *entire* expected signature is known in closed form, isolating engine correctness from anything financial.
Setup: simulate driftless unit Brownian `W` (`r=0`, `σ=1`, `x0=0` on the *value* channel — i.e. simulate `W` directly, not GBM), time-augment to `(t, W_t)`, level `N=3`.
Targets (a few, from `exp_⊗(T·ξ)` with `T` the horizon): `E[S_{(0)}] = T`, `E[S_{(1)}] = 0`, `E[S_{(0,0)}] = T²/2`, `E[S_{(1,1)}] = T/2`, `E[S_{(0,1)}] = E[S_{(1,0)}] = 0`.
Check: `max |MC − analytic_gaussian|` over all coordinates ≤ level 3 is below tolerance.
Tolerance: dominated by MC error; with `n_paths = 1e5`, grid `≈ 250`, expect `< 5e-3`. (State both error sources in the report.)

**T2 — Lead-lag area = quadratic variation.**
Why: the lead-lag embedding is what will let trading P&L (Itô integrals) be read off the signature in the hedging phase; its defining identity must hold.
Check: for several simulated price series, `mean( |area − ½·QV| / (½·QV) )` is small, where `area` is the antisymmetric level-2 coordinate of `sig(lead_lag(series), 2)` and `QV = Σ_k (x_{k+1}−x_k)²`.
Tolerance: `< 1e-2` (it is exact in principle; residual is float/interpolation noise).

**T3 — Monte-Carlo expected-signature convergence.**
Why: quantifies the (benign, because stationary) estimation error and confirms it behaves as theory says — falling like `1/√n`, independent of grid resolution.
Check: compute `‖MC(n) − analytic‖` (Brownian driver, level 3) for `n ∈ {1e2, 1e3, 1e4, 1e5}`; the log-log slope vs `n` is `≈ −0.5`.
Tolerance: fitted slope in `[−0.6, −0.4]`.

**T4 — Forward price vs closed form `X_0 − K·e^{−rT}`.**
Why: the end-to-end check — covector assembly, coordinate indexing, discounting, and the expected signature must combine to the textbook forward value.
Setup: GBM with `r > 0`, some `σ, T, x0, K`.
- *Analytic pipeline:* feed the level-1 price coordinate `x0·(e^{rT}−1)` through `price(...)`; must equal `x0 − K·e^{−rT}` to floating-point tolerance (`< 1e-10`). (This validates the *wiring*: covector + indexing + discounting.)
- *Monte-Carlo pipeline:* feed the MC level-1 price coordinate through `price(...)`; must equal `x0 − K·e^{−rT}` within the reported MC band (`≈ 2·SE`).

---

## 4. Runtime report (clear and motivated)

`report.py` runs all four checks and prints a plain-text report. Every block states its purpose in one line before the numbers, so a reader who has not seen the code understands *why* each check exists. Target shape:

```
=== Signature Pricing Core — Validation Report ===
params: x0=1.0  r=0.03  sigma=0.2  T=1.0  n_steps=250  n_paths=100000  level=3  seed=0

[T1] Signature engine vs Fawcett  (time-augmented Brownian motion)
  why: the only process with a fully known expected signature; checks the engine itself.
  E[∫dt]    1.0000  target 1.0000  |err| 0.0e+00
  E[∫dW]    0.0013  target 0.0000  |err| 1.3e-03
  E[(2,2)]  0.5006  target 0.5000  |err| 6.0e-04
  max coord error (≤lvl 3): 2.4e-03   [MC n=1e5, grid=250]            PASS

[T2] Lead-lag area = 1/2 * quadratic variation
  why: the embedding that will carry Ito trading P&L in the hedging phase.
  mean relative error: 9.1e-04                                        PASS

[T3] Monte-Carlo convergence of E[S]
  why: confirms estimation error decays like 1/sqrt(n), not with grid size.
  n=1e2 err=...   n=1e3 err=...   n=1e4 err=...   n=1e5 err=...
  log-log slope: -0.50                                                PASS

[T4] Forward price vs closed form  X0 - K*e^{-rT}
  why: end-to-end — covector, indexing, discounting, expected signature combine correctly.
  closed form     : 0.970446
  analytic pipe   : 0.970446   |err| 0.0e+00                          PASS
  monte-carlo pipe: 0.970402   |err| 4.4e-05   (MC band 1.1e-04)      PASS

All checks passed. Core is trusted.
```

Exit non-zero if any check fails (so it can gate CI later).

---

## 5. Project layout (minimal)

```
sigcore/
  __init__.py
  gbm.py
  embed.py
  signature.py
  expected_signature.py
  price.py
tests/
  test_engine.py        # T1
  test_leadlag.py       # T2
  test_convergence.py   # T3
  test_forward.py       # T4
report.py               # runs all checks, prints the motivated report, exits non-zero on failure
requirements.txt        # numpy, iisignature, pytest
README.md               # one paragraph: what this validates and how to run it
```

---

## 6. Non-goals — do **not** build these (deferred to later phases)

- Hedging optimisation, loss/shortfall covectors, shuffle products, objective assembly.
- Risk-profile polynomials of any kind.`
- Multi-asset / portfolio joint signatures.
- Heston or any model beyond GBM.
- Web app, plotting, charts, notebooks.
- Log-signatures, signature kernels, randomised signatures.
- Signature *scaling/normalisation*: not needed for a level-1 forward read-off. (It **will** be needed once the dependency matrix appears in the hedging phase; leave a one-line `# TODO(hedging): scale + ridge` where the expected signature is consumed, and stop there.)

If a task seems to need one of these, it belongs to a later phase — flag it and do not implement it.

---

## 7. Numerical notes and gotchas

- **`iisignature` omits the empty word.** Its `sig` returns levels `1..N` only. The empty-word coefficient of the forward covector (`x0 − K`) must be added by hand in `price.price`; that is why `forward_covector` returns it separately.
- **Word ordering must match.** `tensor_exp` output must be flattened in `iisignature`'s convention (level by level; within a level, lexicographic over channel indices) so T1's coordinate-by-coordinate comparison is aligned. Verify the ordering against a tiny hand-checked case (e.g. level ≤ 2, 2 channels) before trusting T1.
- **Two error sources in T1/T3.** Discrepancy from analytic = Monte-Carlo error (`1/√n_paths`) **plus** time-discretisation error (finite grid). Keep tolerances generous and report `n_paths` and `n_steps` alongside every error figure so the two are not confused.
- **Reproducibility.** All randomness flows from a single `seed`; the report prints it.
- **GBM and the Brownian driver share randomness.** GBM is a deterministic transform of the same Gaussian increments used for `W`; you may simulate the increments once and derive both, but this is an optimisation, not a requirement.
- **Tolerances are statistical, not exact**, except the analytic-pipeline arm of T4, which is exact to floating point and therefore catches wiring bugs precisely.
