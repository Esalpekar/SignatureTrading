# Signature Pricing Core — Test Suite Spec (v1: Comprehensive Validation)

> Audience: a coding agent extending the existing `sigcore/` harness (gbm, embed, signature, expected_signature, price).
> Goal: validate **every layer** of the plumbing at depth — engine, embedding, expected-signature estimation, pricing — not just the trivial level-1 forward; **quantify the estimation error per level**; and bring in Heston. When this suite passes, the plumbing is trustworthy above level 1.
> **Build only what these tests require:** a Heston simulator, small helpers (shuffle, log-embedding, Asian covector, closed-form targets), and the test files. No hedging, no risk polynomials, no portfolio (those get their own specs later).

---

## 0. Test philosophy — read first

Use the **strongest available evidence** for each layer, in this priority order. A test that relies on a weaker level when a stronger one was available is a weak test; push every check as far up as the math allows.

1. **Exact algebraic identities.** Hold per-path to machine precision, independent of any model or statistics. Bedrock — if these fail, nothing downstream is meaningful.
2. **Closed-form oracles.** Compare to an analytic value wherever one exists. Two oracles unlock depth (§1): the **log-coordinate tensor exponential** (GBM's expected signature at *all* levels) and the **arithmetic-average Asian forward** (a closed-form *path-dependent* price).
3. **Internal consistency + variance control.** Only where no oracle exists (Heston's higher coordinates): split-sample agreement and bounded relative standard error.

Report numbers, not just pass/fail. The point of this suite is partly to *characterise* the estimation error, not merely to assert it is small.

---

## 1. Closed-form facts the suite uses (do not re-derive)

- **Straight line `a → b`** (any dimension): the signature is the tensor exponential of the increment. Level-`k` term `= (b − a)^{⊗k} / k!`. Exact.
- **GBM in log-coordinates.** `Y_t = log X_t` is arithmetic Brownian motion with drift: `dY = (r − σ²/2)dt + σ dW`. The time-augmented path `(t, Y_t)` (channel 0 = time, channel 1 = logX) has expected signature
  `E[S_{0,T}] = exp_⊗(T·ξ)`, `ξ = e₀ + (r − σ²/2)·e₁ + (σ²/2)·(e₁ ⊗ e₁)`.
  This is analytic at **all** levels. (Validate against the existing `signature.tensor_exp`.)
- **Arithmetic-average forward.** With channels (0 = time, 1 = price), the average `A = (1/T)∫₀ᵀ X_t dt = X₀ + (1/T)·⟨e_{(1,0)}, S⟩` (the word "price then time"). The forward paying `A − K`:
  - covector `f = (X₀ − K)·∅ + (1/T)·e_{(1,0)}`,
  - price `= e^{−rT}·(E[A] − K)`, `E[A] = X₀·(e^{rT} − 1)/(rT)`.
  - **Removable singularity at `r = 0`:** `E[A] → X₀`, so price `→ X₀ − K`. The code must use the limit when `|r|` is below a small epsilon, not divide by zero.
- **Heston integrated variance.** For `dv = κ(θ − v)dt + ξ√v dW₂`,
  `E[∫₀ᵀ v_t dt] = θ·T + (v₀ − θ)·(1 − e^{−κT})/κ`.
  Since the quadratic variation of `log X` over `[0,T]` is `∫₀ᵀ v_t dt`, the **mean lead-lag area** of the log-price path equals **½** of this. (Level-2 Heston check against closed form.)
- **Forward (price coords):** `X₀ − K·e^{−rT}` (already validated in v0; retained as C1).

---

## 2. The suite

For every test: **what** it checks, **why**, the **oracle/target**, the **embedding** it uses, and the **tolerance**.

### Category A — Algebraic identities (exact, per-path, machine precision, model-free)

These need no model and no Monte Carlo. Run each on several deterministic and random paths (mix of 2- and 3-channel, uniform and irregular grids). Tolerance for all: `< 1e-10` (allow float roundoff in tensor arithmetic).

- **A1 Straight-line closed form.** Signature of a linear segment `a→b` equals `Σ_k (b−a)^{⊗k}/k!` up to level `N`. *Why: the most basic iterated-integral computation; if this is wrong, everything is.*
- **A2 Shuffle identity.** For words `u, v` (lengths `p, q` with `p+q ≤ N`), `⟨S,u⟩·⟨S,v⟩ = ⟨S, u⧢v⟩`, where `u⧢v` is the shuffle (sum of interleavings with multiplicity). *Why: the defining group-like property; also pre-validates the shuffle machinery the hedging phase will depend on.* (Implement a small `shuffle(u, v) -> Counter[word]`.)
- **A3 Chen's identity.** Split a path at an interior index into `P₁, P₂`; then `sig(P₁⋆P₂) = sig(P₁) ⊗ sig(P₂)` (truncated tensor product). *Why: validates the concatenation structure.*
- **A4 Inverse / time reversal.** `sig(reverse(P))` is the tensor inverse of `sig(P)` (their truncated product is the identity, `∅`). *Why: validates antipode/sign structure.*
- **A5 Reparameterisation.** For a path with **time NOT a channel**, resampling the same geometric trajectory on a different (increasing) time grid leaves the signature unchanged. With **time augmented**, the signature *does* change. *Why: confirms reparam-invariance and that time-augmentation deliberately breaks it.*
- **A6 Level-1 = increments.** `⟨S, (i)⟩ = X^i_T − X^i_0` exactly for each channel `i`. *Why: validates channel→index mapping.*
- **A7 Homogeneity.** Scaling channel `i` by `λ` scales each coordinate by `λ^{(count of i in the word)}`; scaling all channels by `λ` scales level-`k` by `λ^k`. *Why: validates degree structure; relevant to the scaling/conditioning work later.*
- **A8 Degenerate paths.** Constant path → only `∅` is nonzero (all levels `1..N` are zero). Single-step path → matches A1. *Why: boundary behaviour.*

### Category B — Expected-signature estimation accuracy (statistical, vs the all-level oracle)

Embedding: **time-augmented log-price GBM** `(t, log X_t)`, so the oracle `exp_⊗(T·ξ)` from §1 applies at all levels.

- **B1 All-level oracle comparison.** Compare the MC expected signature to `exp_⊗(T·ξ)` for **every** coordinate up to level `N=4`. Primary gate: per-level **max absolute error** below a per-level threshold (errors grow with level; thresholds are empirical — do a one-time calibration run and record them, e.g. roughly `1e-3, 5e-3, 2e-2, 8e-2` for levels 1–4 at `n_paths=1e5`, `n_steps≥250`). Secondary diagnostic: fraction of *stochastic* coordinates within `4·SE` of the oracle should be ≳ 0.95. *Why: this is the real depth test and closes the GBM coverage gap — it certifies the higher-level signature structure, not just level 1.*
- **B2 Per-level error + standard-error table.** Report, per level: max and mean `|MC − oracle|`, and the mean coordinate standard error (`std/√n`). *Why: turns "the error is low" into an auditable table; shows the error climbing with level.*
- **B3 Sampling-error scaling.** For a representative coordinate at each level, confirm the empirical SE scales `~1/√n_paths` (e.g. `SE(4n)/SE(n) ≈ 1/2`). *Why: confirms the estimator behaves as theory predicts; isolates sampling error from bias.*
- **B4 Discretisation bias.** Hold `n_paths` very large (so sampling error is negligible) and vary `n_steps`; show the residual error shrinks as the grid refines. *Why: exposes the discretisation floor separately — the thing that flattened the v0 convergence slope below −0.5. Disentangling B3 and B4 explains that slope.*
- **B5 Seed robustness.** Run B1 across ≥ 20 seeds; report the distribution (mean, max) of the per-level max error; require **every** seed to pass the B1 thresholds. *Why: the v0 T1 passed at one seed with 98% of its budget spent — a single-seed pass is not a pass.*

### Category C — Pricing correctness (vs closed form)

- **C1 Forward (level 1).** Price coords; target `X₀ − K·e^{−rT}`. Analytic-pipeline arm exact (`< 1e-10`); MC arm within `4·SE`. *(Retained from v0; note it only exercises level 1.)*
- **C2 Arithmetic-average Asian forward (level 2).** Price coords; covector and closed form from §1. Analytic arm (feeding the oracle level-2 coordinate `E[∫(X_t−X₀)dt]`) within `< 1e-6`; MC arm within `4·SE`. **The key non-first-order pricing test** — it certifies that `⟨f, E[S]⟩` is correct at level 2 against a path-dependent closed form.
- **C3 Pricing linearity.** `price(αf + βg) = α·price(f) + β·price(g)` to `< 1e-10`. *Why: the inner product is linear; this guards the covector arithmetic and supports later semi-static / portfolio combinations.*
- **C4 Pricing edge cases.** (i) `r = 0`: forward → `X₀ − K`, Asian → `X₀ − K` (tests the removable singularity path). (ii) `K = 0`: forward → `X₀`. (iii) Pure-constant "bond" payoff `f = 1·∅` → `e^{−rT}`. *Why: validates empty-word handling and the `r=0` limit.*

### Category D — Heston (consistency + partial closed forms)

Add `heston.py` (see §7). Use Feller-satisfying base parameters for D1–D4; a Feller-violating set for D5.

- **D1 Identities on Heston paths.** A2 (shuffle) and A3 (Chen) hold per-path to `< 1e-10` on Heston paths. *Why: model-free identities must survive a rougher, stochastic-vol path; cheap, strong confirmation the engine handles Heston correctly.*
- **D2 Martingale / level-1.** MC `E^Q[X_T] = X₀·e^{rT}` within `4·SE`. *Why: under Q the discounted price is a martingale regardless of the vol process; basic drift check.*
- **D3 Integrated variance via lead-lag (closed form).** Embedding: lead-lag of **log-price**. MC mean lead-lag area within `4·SE` of `½·(θT + (v₀−θ)(1−e^{−κT})/κ)`. *Why: a genuine higher-level (level-2) Heston check against an oracle, and it exercises the lead-lag embedding the hedging phase needs.*
- **D4 Higher-order variance control.** Embedding: time-augmented log-price. Split the sample in half; the two half-sample expected signatures must agree, per coordinate, within their combined SE. Additionally, require the **relative standard error** of every level-3 coordinate to be below a threshold (e.g. `< 10%`) at `n_paths = 1e5`. *Why: this is the "Heston MC'd into a signature with low variance in higher-order terms" requirement, made into a pass/fail number, in the absence of a full oracle.*
- **D5 Feller / positivity.** With Feller-violating parameters (`2κθ < ξ²`), assert the simulated variance path never goes negative (the scheme must clamp). *Why: a naive Heston discretisation produces negative variance and silently corrupts everything downstream.*

### Category E — Embedding & numerical edge cases

- **E1 Channel order.** Assert time is channel 0 everywhere; verify via A6 (time coord `= T`, price coord `= X_T − X₀`). *Why: a swapped channel order corrupts covector indexing silently.*
- **E2 Lead-lag shape/alignment.** Assert the lead-lag output has the expected length and that lead leads lag by one sample; correctness of the area is already T2/D3. *Why: off-by-one in the staircase is a classic bug.*
- **E3 Irregular grid.** All Category-A identities still hold on a non-uniformly sampled path (`< 1e-10`). *Why: signatures are built for irregular sampling; confirm the implementation does not secretly assume a uniform grid (matters for real data later).*
- **E4 Determinism.** Same seed → identical arrays (`allclose` with zero tolerance, or exact equality). *Why: reproducibility of every reported number.*
- **E5 Magnitude sanity.** No `inf`/`nan` in any signature; flag coordinates whose magnitude is pathological for the chosen scale. *Why: factorial growth/decay across levels can overflow/underflow; this is the early-warning that the deferred scaling will be needed for hedging.*

---

## 3. Edge-case checklist (consolidated)

The plumbing must satisfy all of:
`r = 0` (no discounting; Asian removable singularity) · `K = 0` (forward → spot) · pure-constant payoff (bond) · constant/zero-movement path · single-step path · irregular (non-uniform) time grid · channel-order correctness · lead-lag off-by-one/length · Heston Feller violation (variance positivity) · multiple seeds (no single-seed passes) · determinism (seed → identical output) · no `inf`/`nan`, no pathological magnitudes · large `n_paths` with coarse vs fine `n_steps` (separating sampling error from discretisation bias).

---

## 4. Reporting (the estimation-error answer, with numbers)

Extend `report.py`. In addition to per-test pass/fail, print:

- a **per-level error table** for B1/B2: `level | n_coords | max|err| | mean|err| | mean SE | within-4SE %`;
- the **B3/B4 breakdown**: sampling-error scaling vs `n_paths`, and bias vs `n_steps`, side by side, so the two floors are visibly separated;
- a **Heston block** for D3/D4: integrated-variance check, and the relative-SE-by-level summary with the worst level-3 coordinate named;
- a one-line **purpose** before each block (as in v0), and a non-zero exit if any gate fails.

The report should let a reader answer, in numbers: how large is the signature-estimation error at each level, how does it split between sampling and discretisation, and how stable are Heston's higher-order coordinates.

---

## 5. Acceptance thresholds (summary)

- Category A, C3, C4, D1, E3, E4: `< 1e-10` (exact identities / linear algebra).
- C1/C2 analytic arms: `< 1e-6`; MC arms: within `4·SE`.
- B1: per-level absolute thresholds (calibrate once, record in the README); secondary within-4SE ≳ 95%.
- B3: SE scaling ratio within ±15% of `1/√(factor)`. B4: monotone decrease in bias with `n_steps`.
- B5: all ≥ 20 seeds pass B1.
- D2, D3: within `4·SE`. D4: level-3 relative SE `< 10%`; half-sample agreement within combined SE. D5: zero negative variance values.

Thresholds set with `4·SE` are statistical; the exact-identity and analytic-pipeline thresholds are deterministic and catch wiring bugs precisely.

---

## 6. Non-goals (unchanged)

No hedging optimisation, no loss/shortfall covectors beyond what pricing needs, no risk-profile polynomials, no portfolio/multi-asset, no signature scaling/ridge (still a deferred `# TODO(hedging)`), no web app or plots. **Exception, by design:** the shuffle product and Chen's identity *are* implemented and tested here, because they are properties of the existing signature engine and pre-validating them de-risks the hedging phase.

---

## 7. Implementation notes and gotchas

- **Log vs price coordinates — be explicit per test.** Use **log-price** embedding for the all-level oracle (B1–B5), for Heston integrated variance (D3), and for the variance-control checks (D4); use **price** embedding for the forward and the arithmetic Asian (C1, C2) because their covectors are clean there. Mixing these up is the most likely source of a confusing failure.
- **The `r = 0` Asian singularity.** `E[A] = X₀(e^{rT}−1)/(rT)` has a removable singularity at `r = 0`; implement the limit `X₀` for `|r| < 1e-8`.
- **Heston scheme.** Use full-truncation Euler (clamp `v` at 0) or the Andersen QE scheme; either keeps variance non-negative (required for D5). Return both price and variance paths. Correlate the two Brownian drivers with `ρ`.
- **Variance control (D4).** Use a simple sample split (or bootstrap) rather than anything fancy; the test is agreement within combined standard error plus a relative-SE bound.
- **Shuffle helper.** A2/D1 need `shuffle(u, v)`: recursively, `shuffle(au, bv) = a·shuffle(u, bv) + b·shuffle(au, v)`, base cases the empty word. Return a multiset of words; this is also the exact primitive the hedging phase will reuse.
- **Reuse existing modules.** New code is limited to: `heston.py`; small helpers (`shuffle`, a log-embedding wrapper, the arithmetic-Asian covector, and the closed-form target functions); and the test files. Do not duplicate `signature.tensor_exp` — reuse it for the oracle. Do not build the full GBM (price-coordinate) expected signature; the log-coordinate oracle is sufficient and cleaner.
- **Word ordering.** All coordinate extraction must use the existing `signature.word_index`, consistent with `iisignature` (level-by-level, lexicographic, empty word omitted), so oracle comparisons stay aligned.