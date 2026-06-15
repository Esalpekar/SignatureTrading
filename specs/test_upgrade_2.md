# Signature Pricing Core — Zero-Trust Audit & Test Additions (v2)

> Audience: a coding agent extending `sigcore/`.
> Premise: **the green report proves less than it looks.** A passing test certifies a property only if (a) it actually ran and (b) the payoff/path it used could have *exposed* the bug. This document lists what the current suite does not establish and what to add. Treat the report as a subset until proven otherwise.
> Guiding question for every test: *what bug would leave the whole report green?*

---

## A. The report is a subset — verify before trusting (do this first)

`python report.py` prints **T1–T4, B, D**. It does **not** print, and this run did not execute:

- **Category A** — the algebraic identities (straight-line A1, shuffle A2, Chen A3, inverse A4, reparam A5, level-1=increment A6, homogeneity A7, degenerate A8). These are the *strongest* evidence: exact, per-path, model-free.
- **C2** — the arithmetic-average Asian forward, the **only** path-dependent *pricing* test. The report's B block validates the expected signature at depth; it does **not** validate that the pricing inner product yields a correct path-dependent price. T4 is level-1 only.
- **C3, C4** — pricing linearity and the `r=0`/`K=0`/bond edge cases.
- **D1, D2, D5** — identities on Heston paths, the martingale check, and Feller/positivity.
- **Category E** — channel order, lead-lag shape, irregular grid, determinism, magnitude.
- **B5** — the seed sweep. T1 is at ~98% of its tolerance at seed 0 with no robustness data.

**Why this is dangerous, not just untidy.** A1 (straight line) and A6 (level-1 = increment) are the *independent* checks on `word_index` and `tensor_exp`. The B oracle compares MC to `tensor_exp` using `word_index` on both; a consistent indexing/labeling bug could make them agree while both are mislabeled. Only A pins the labels to ground truth. If A is not actually running, B's depth result is not trustworthy.

**Actions (gate before any further build):**
1. Run the **full `pytest` suite**, not `report.py`. Confirm every test above **exists and passes**.
2. If any are missing, implement them per the v1 spec before proceeding. Priority order: A1, A6, A2 (these guard everything else), then C2, then the rest.
3. Wire a one-line pass/fail for **each** category into `report.py` so a single `report.py` run can never again hide a missing category. The report must enumerate every test group, even those whose detail lives in `pytest`.

---

## B. Gaps in coverage of **existing** code — add these now

### U — Universality / truncation convergence (the most important missing test)
**What is untested.** Every payoff priced so far (forward, Asian) is *exactly* a signature coordinate, so its truncation error is identically zero. The method's entire claim — that a linear functional of the truncated signature *approximates* an arbitrary continuous payoff, error → 0 as level `N` → ∞ — has never been exercised. *Bug that survives:* a silently-too-shallow truncation, a non-convergent representation, a wrong notion of which payoffs are representable — all pass, because only exactly-representable payoffs were tested.

**Test.** Use a **non-polynomial** payoff with a closed-form price under GBM: the power claim `payoff = X_T^p`, `p` non-integer (e.g. `p = 1.5` and `p = 0.5`), price `= e^{−rT}·X₀^p·exp(p(r−σ²/2)T + ½p²σ²T)` (lognormal moment).
Build its covector at each truncation level `N` from the Taylor expansion of `x ↦ x^p` about `X₀`: the term `(X_T − X₀)^k` is the shuffle power of the level-1 price coordinate, so the covector is `Σ_{k≤N} c_k · (price-letter)^{⧢k}` with `c_k` the Taylor coefficients. Price via `⟨f, E^Q[S]⟩`.
**Assert:** the pricing error vs the closed form **decreases monotonically** as `N` grows (report it: `N=1,2,3,4,5`), and reaches a stated tolerance by `N=5`. *(No regression needed — the covector is constructed analytically. This isolates the truncation/approximation property from the fitting machinery.)*
**Stretch (optional, vs MC not closed form):** a path-dependent non-polynomial payoff (e.g. `exp` of the arithmetic average); regress or Taylor-build its covector and show `⟨f, E[S]⟩` converges to the **direct Monte-Carlo** price of the payoff as `N` grows.

### I — Itô recovery with a **state-dependent** integrand
**What is untested.** T2/D3 verify lead-lag area `= ½·QV` — but QV is the Itô integral of a *constant* integrand. The hedging P&L is `∫θ_t dX_t` with `θ` varying along the path. *Bug that survives:* a lead-lag that nails the area but mishandles a varying integrand → correct QV, wrong hedge P&L.

**Test.** Per path, recover `∫₀ᵀ X_t dX_t` from the lead-lag signature (the level-2 coordinate corresponding to "integrate price against price"). Compare to the exact Itô value `½(X_T² − X₀²) − ½·QV`, where `QV = Σ_k (X_{t_{k+1}} − X_{t_k})²`.
**Assert:** per-path agreement to `< 1e-8` (it is an exact identity in the discrete lead-lag construction; residual is float noise). This is the real test of the mechanism the hedge will run on. *(Specify the property, not the exact word index; let the implementation locate the coordinate, as with T2.)*

### Foresight tests (cheap, build now)
- **Conditioning probe.** For a toy single-asset case (level-1 and level-2 trading covectors), assemble the dependency/Gram matrix `A_{wv} = ⟨(w·trade) ⧢ (v·trade), E[S]⟩` (needs only the shuffle from A2 and `E[S]`) and **report its condition number by truncation level**. No hard pass/fail yet — this is the early-warning instrument for the estimation wall that the hedging phase will hit. Watching `cond(A)` blow up with level *now* prevents it being a surprise later.
- **B5 seed sweep — make it run.** Re-run the B all-level comparison across ≥ 20 seeds; report the distribution (mean, max) of per-level max error; require **every** seed under threshold. Single-seed passes do not count.
- **E5 magnitude — make it run.** Assert no `inf`/`nan` and flag pathological coordinate magnitudes; this is the precursor signal that the deferred signature scaling will be needed before hedging.

---

## C. Gaps that test **future** code — mandatory gates, not patches

These validate capabilities not yet built. Do **not** build the corresponding phase without these tests written alongside it.

### M — Multi-asset / cross-asset coordinates (gate for the portfolio phase)
**What is untested.** Everything is single-asset (one price + time). The cross-asset, mixed-letter coordinates — the off-diagonal correlation, co-skewness, the pairs-trading structure — are validated by nothing. *Bug that survives:* any error in how the joint signature mixes channels, any sign error in cross-asset Lévy area.
**Test (when portfolio code exists).** Two correlated Brownian motions `(W¹, W²)` with correlation `ρ`, time-augmented to `(t, W¹, W²)`. The joint expected signature is `exp_⊗(T·ξ)` with diffusion block `[[1, ρ],[ρ, 1]]`, i.e. `ξ = e₀ + ½(e₁⊗e₁ + e₂⊗e₂ + ρ(e₁⊗e₂ + e₂⊗e₁))`. Compare MC to this oracle at all levels; in particular the symmetric cross coordinate `E[⟨(1,2),S⟩ + ⟨(2,1),S⟩]` must equal `ρT`. Also run A2/A3 (shuffle, Chen) on the 3-channel paths.

### R — Regression covector fitting (gate for the arbitrary-payoff phase)
**What is untested.** The "arbitrary payoff" path fits the covector by least squares (Algorithm 1). Untested, and likely unbuilt.
**Test (when regression code exists).** Regress a payoff whose covector is known in closed form (forward, Asian) onto the signature; **recover that covector** to a stated tolerance; report the regression **R²** and the **condition number** of the design matrix. Then confirm the regressed covector reproduces the closed-form price.

---

## D. The zero-trust principles (carry these forward)

- A test that uses a payoff which is an **exact** signature coordinate cannot validate the **approximation** — it has nothing to approximate.
- A report that prints a **subset** cannot certify the **whole**; every category must surface in the report, with detail in `pytest`.
- An oracle (`tensor_exp`) compared only against an estimator that **shares its indexing** can agree while both are mislabeled; an **independent** ground-truth check (straight-line, level-1 = increment) is required.
- A mechanism tested only in its **degenerate case** (constant integrand → QV) is not validated for its **general case** (state-dependent integrand → hedge P&L).
- A single-**seed** pass is not a pass.
- Validate the capability **when you build it**: multi-asset and regression tests are written *with* the code they test, never after.

---

## E. Non-goals (unchanged)

No hedging optimisation, no risk-profile polynomials, no portfolio *implementation* yet (only its test gate, §M), no signature scaling/ridge beyond the conditioning probe, no web app. The shuffle and Chen identities remain in scope as engine properties (and the shuffle is now also used by the conditioning probe).