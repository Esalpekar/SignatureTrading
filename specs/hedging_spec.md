# Signature Hedging Phase — Implementation Spec (v1)

> Audience: a coding agent extending the validated `sigcore/` core.
> This phase turns the validated primitives into a working hedger. Every component below ships **with its test gate in the same step** — the zero-trust principle from the audit, applied from the start rather than after. Build a component and its gate together; do not proceed past a red gate.
> **Scope is deliberately narrow** (see §6): closed-form-covector payoffs (forward, Asian), single asset, mean-variance plus one *convex* asymmetric penalty. No regression, no multi-asset, no non-convex penalties.

---

## 0. What this phase is, and the one decision baked in

The core proved the pipeline computes expected signatures and prices linear claims correctly, at depth. This phase adds the **hedge**: given a liability, find the self-financing trading strategy that minimises a chosen risk functional of the shortfall, and validate it.

**Measure decision (baked in, overridable).** Everything runs under the **risk-neutral measure** (drift `= r`). Consequences, all deliberate: the premium `p₀` is the Q-price from the core, which makes the shortfall a natural mean-zero reference; the complete-market forward is perfectly replicable, giving a clean oracle. **Incompleteness — the thing that makes the risk profile do anything — comes from Heston (stochastic vol, unhedgeable with the underlying alone) or from coarse rebalancing, not from a drift gap.** The physical-measure ("realistic P&L") evaluation is a non-goal here; folding it in would corrupt every oracle below.

**Build on, do not rebuild:** `gbm`, `heston`, `embed` (time-augment + lead-lag), `signature` (including the `shuffle` validated by A2), `expected_signature`, `price`, the `tensor_exp` oracle, and the conditioning probe. The trade-letter/P&L wiring **is** the mechanism validated by test **I** (Itô recovery); this phase consumes that guarantee.

---

## 1. The hedging mathematics (implement exactly these)

Single asset. The enlarged path is **time-augmented + lead-lag of price**; `S` denotes its (running or terminal) signature. Strategy depth `d_strat`, expected-signature truncation `N_sig`, penalty degree `q`.

- **Strategy.** Position at time `t` is `θ_t = ⟨ℓ, S_{0,t}⟩` — a covector `ℓ` against the *running* signature. A feedback rule, not a schedule.
- **Trading P&L.** `∫₀ᵀ θ_t dX_t = ⟨ℓ·a, S_{0,T}⟩`, where appending the **trade letter** `a` (the lead-lag price channel) yields the **Itô** integral. This is exactly the mechanism test I validated; reuse that channel.
- **Loss covector.** `𝓛 = f − p₀·∅ − ℓ·a`, with `f` the payoff covector (forward or Asian, closed-form from the core), `p₀` the premium (= Q-price). Realised shortfall `L = ⟨𝓛, S⟩` — linear in the signature, **affine in the strategy coefficients `ℓ`**.
- **Objective.** `R(ℓ) = E[P(L)] = ⟨P^⧢(𝓛), E[S]⟩`, where `P^⧢(𝓛) = Σ_k c_k · 𝓛^{⧢k}` (shuffle powers) and `E[S]` is the expected signature from the core. The shuffle turns "expected polynomial of a random shortfall" into one contraction against `E[S]`.
- **Mean-variance (`P(x)=x²`).** `R(ℓ) = const − 2·b·ℓ + ℓᵀ A ℓ`, with
  `A_{wv} = ⟨(w·a) ⧢ (v·a), E[S]⟩` (the **dependency/Gram matrix** — the same `A` as the conditioning probe),
  `b_w = ⟨(f − p₀∅) ⧢ (w·a), E[S]⟩`.
  Optimum: solve `(A + λI)·ℓ = b` (ridge `λ`). For `λ=0`, `ℓ* = A⁻¹b`.
- **Convex asymmetric penalty (`P(x) = x² + γx³ + δx⁴`, `δ ≥ 3γ²/8` for convexity).** `R(ℓ)` is degree-4 and **convex** in `ℓ` (convex-composed-with-affine, then expectation). Minimise by Newton: gradient `∇R` (degree 3 in `ℓ`) and Hessian `H(ℓ)` (degree 2 in `ℓ`) are analytic contractions of shuffle powers with `E[S]`. Mean-variance is the special case where `H` is constant and Newton converges in one step.
- **Truncation–degree coupling (hard constraint).** `𝓛^{⧢k}` reaches words of length `q·d_strat`, so the objective requires `E[S]` to level `N_sig ≥ q·d_strat`. With the core's level-4 `E[S]`: mean-variance (`q=2`) allows `d_strat ≤ 2`; the quartic (`q=4`) allows `d_strat ≤ 1`. Going deeper means recomputing `E[S]` to higher levels.
- **Conditioning.** `A` is ill-conditioned (probe: ~100×/level). Apply **level-wise signature scaling** (a diagonal change of basis equalising coordinate magnitudes — e.g. divide level-`k` coordinates by `k!` or by their empirical scale) before assembling `A` and `b`, and **ridge**-regularise the solve. Scaling must be undone when reading off `θ_t`.

---

## 2. Modules and behaviour (each with its gate — build together)

- **`hedge_objective.py`** — assemble `A`, `b` (mean-variance); and `P^⧢(𝓛)`, `∇R`, `H` (general convex). *Gate H0 below.*
- **`scaling.py`** — level-wise scaling and its inverse, applied consistently to `ℓ`, `f`, `E[S]`. *Gate H4.*
- **`solve.py`** — mean-variance ridged linear solve; convex Newton (with line search/damping for safety). *Gates H1, H5.*
- **`strategy.py`** — evaluate `θ_t = ⟨ℓ*, S_{0,t}⟩` along a path from its running signature. *Gate H8.*
- **`pnl.py`** — simulate the shortfall distribution: apply `θ_t` on a (possibly coarse) grid, accumulate `θ·ΔX`, form `L = f − p₀ − P&L` per path; report variance, percentiles, CVaR. Reuse `heston` for incompleteness and expose a `rebalance_steps` knob for coarse-rebalancing incompleteness. *Gates H3, H6.*

No other new modules. Reuse the core for everything else.

---

## 3. Test gates (the suite — interleaved, baked in)

Ordered by dependency. Each states what it checks, **why**, the oracle, and the tolerance. Wire every gate into the `report.py` inventory so none can hide (the v2 lesson). Statistical gates must be seed-robust (run ≥ 20 seeds or set generous margin) — the suite is the authoritative gate and a flaky gate is worse than none.

- **H0 — Objective cross-check (BEDROCK).** The shuffle-assembled objective `R(ℓ) = ⟨P^⧢(𝓛), E[S]⟩` must equal a **direct Monte-Carlo** estimate of `E[P(L)]` (simulate paths, apply `θ_t = ⟨ℓ, S_{0,t}⟩`, form `L`, average `P(L)`), for several random `ℓ`, for both `P=x²` and the quartic. Within `4·SE`. *Why: this independently validates the entire assembly — loss covector, trade-letter append, shuffle, `A`, `b` — against a simulation that shares none of that code. If H0 fails, nothing else is meaningful.*
- **H1 — Complete-market replication oracle.** GBM, **`r=0`**, forward, mean-variance, analytic `E[S]`, `λ≈0`. Assert `ℓ*` ≈ the constant strategy `θ=1` (within tolerance) and `E[L²] ≈ 0` (machine precision — at `r=0` the constant-1 hedge gives `L=0` exactly per path). *Why: the one case with a known exact hedge; certifies the solve recovers the true replicating strategy and drives residual risk to zero.*
- **H2 — Level-0 hedge = regression coefficient.** The depth-0 (constant) mean-variance hedge ratio must equal `Cov(F, ΔX)/Var(ΔX)` (with `p₀ = E[F]`). Oracle: a **direct-MC** estimate of that covariance ratio (independent of the signature path); for the forward it must also equal `1`. Forward and Asian. Within `4·SE`. *Why: connects the machinery to the textbook hedge and validates `b` and the depth-0 `A`.*
- **H3 — Variance reduction, monotone with depth, plateau at the incompleteness floor.** Under **Heston**, `Var(L)` at `d_strat = 0, 1, 2` must (a) be strictly below the unhedged `Var(F − p₀)`, and (b) decrease with depth, plateauing above zero (the unhedgeable-vol floor). *Why: shows the hedge actually helps, that richer strategies help more, and that incompleteness leaves a genuine residual — the precondition for the risk profile to matter.*
- **H4 — Scaling and ridge.** (a) Scaling reduces `cond(A)` substantially versus unscaled (report both). (b) With MC `E[S]`, the ridged `ℓ*` is stable across ≥ 20 seeds (bounded coefficient variance), whereas un-ridged it is not. (c) With analytic `E[S]`, `λ → 0` recovers the unregularised solution. *Why: the conditioning probe proved the wall exists; this proves the mitigation works and is necessary.*
- **H5 — Convexity / unique optimum.** For the convex quartic, Newton from several distinct starting points converges to the **same** `ℓ*` (within tolerance). Confirm mean-variance is recovered as the one-Newton-step case. *Why: certifies the convex solve has a unique global optimum and is found reliably — the reason non-convex penalties were excluded.*
- **H6 — Asymmetric penalty reshapes the tail (THE headline).** Under **Heston** (or coarse rebalancing), fit the mean-variance hedge `ℓ_MV` and the convex-quartic hedge `ℓ_asym`; apply both to **fresh** paths. Assert: `Var(L_asym) ≥ Var(L_MV)` (it accepts more variance) **and** a left-tail loss measure is smaller for `asym` — specifically `CVaR₉₅(L_asym) < CVaR₉₅(L_MV)` (mean of the worst-5% losses), and the shortfall skewness is more negative. Report both distributions' variance, 95% CVaR, and `P[L > τ]` for a fixed `τ`. *Why: this is the project's premise made checkable — the personalised risk profile demonstrably trades variance for a thinner loss tail. It is meaningless in a complete market, so it MUST run incomplete.*
- **H7 — Truncation–degree coupling enforced.** Requesting `d_strat > ⌊N_sig/q⌋` must raise a clear error (or trigger an `E[S]` recompute), never silently truncate the objective. *Why: a silently dropped shuffle term biases the objective and can break convexity.*
- **H8 — Out-of-sample consistency.** Fit `ℓ*` on `E[S]`; on **fresh** paths, the realised P&L distribution (from `strategy.py` + `pnl.py`) matches the in-sample prediction within `4·SE`. *Why: in the stationary generated-data regime in-sample = out-of-sample, so a mismatch means the feedback evaluation `θ_t = ⟨ℓ, S_{0,t}⟩` is wired wrong.*

**Edge cases (assert):** `ℓ = 0` gives exactly the unhedged shortfall (baseline). `p₀ = price` makes `E[L] ≈ 0` for a good hedge. Asian and forward both run through every gate where a covector is needed. `r = 0` path (H1) and a small `r > 0` path (H0/H3/H6) both exercised.

---

## 4. Reporting

Extend `report.py` with a hedging block, every gate enumerated in the inventory:

- H0 objective cross-check (shuffle vs direct-MC, with SE).
- H1 replication oracle (`ℓ*` coefficients, `E[L²]`).
- H3 **variance-reduction curve**: `Var(L)` by `d_strat`, against the unhedged baseline and the plateau.
- H4 `cond(A)` unscaled vs scaled; ridged-coefficient stability across seeds.
- H6 the **headline numbers**: a side-by-side table of `Var`, `CVaR₉₅`, `P[L>τ]`, and skewness for mean-variance vs the asymmetric hedge, so the variance-for-tail trade is visible in numbers. (The histogram visualisation is the web-app phase, not here — console numbers only, per the core's convention.)

---

## 5. (No new) — reuse and consistency notes

- The trade-letter append reuses the lead-lag channel validated by **test I**; do not introduce a second Itô-recovery path.
- Scaling is a diagonal change of basis: assemble `A`, `b` in scaled coordinates, solve, then map `ℓ*` back before evaluating `θ_t`. A scaling applied inconsistently across `ℓ`, `f`, and `E[S]` is the most likely silent bug — H0 and H1 are the guards.
- The conditioning probe's `A` and this phase's mean-variance `A` are the same object; share the assembly code.

---

## 6. Non-goals (deferred — do not build)

- **Regression / arbitrary payoffs.** Only closed-form covectors (forward, Asian) here. The arbitrary-payoff path (Algorithm 1) and its test gate **R** stay deferred, as you requested.
- **Multi-asset / portfolio.** Single asset only. The multi-asset *engine* oracle (gate **M**, correlated BMs vs `tensor_exp`) is testable against the existing engine and should be done before any portfolio phase — but the portfolio *allocation logic* and its hedging are out of scope here.
- **Non-convex penalties.** Convex `P` only (`δ ≥ 3γ²/8`). No homotopy continuation; the multiple-optimum machinery is excluded by design.
- **Physical-measure / realistic P&L.** Everything under Q; the P-vs-Q performance distinction is deferred.
- **Transaction costs, financing frictions, continuous-time exactness.** Discrete rebalancing only; frictionless.

---

## 7. Implementation gotchas

- **Incompleteness is mandatory for H3 and H6.** Run them under Heston (Feller-satisfying params, the validated simulator) or coarse rebalancing. Running the asymmetric-penalty comparison under fine-grid GBM will show *no* difference between `MV` and `asym` (complete market → both are the perfect hedge) and will look like a bug when it is the math.
- **`r = 0` for H1 only.** The exact-replication oracle needs `r=0` so the constant-1 hedge gives `L=0` with no discounting subtlety. Other gates use a small `r>0`.
- **Ridge selection.** A fixed small `λ` is acceptable for v1; report the solution's sensitivity to `λ`. Do not auto-tune elaborately — minimalism.
- **Newton safety.** Damp/line-search the Newton step for the quartic so it cannot overshoot; convexity guarantees convergence but not that a raw full step is safe far from the optimum.
- **Seed robustness on the authoritative gate.** Any statistical gate (H0, H2, H3, H6, H8) feeds `report.py`'s exit code; give each real margin or a seed sweep so a failure means a bug, not an unlucky draw.