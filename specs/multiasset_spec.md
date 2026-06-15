# Signature Hedging — Multi-Asset Extension (v1)

> Audience: a coding agent extending the validated single-asset hedger.
> Multi-asset is mathematically a clean generalisation: a vector path, one trade letter and one strategy covector per asset, and a **block-structured** dependency matrix. The risk is not the math — it is that the cross-asset coordinates are untested (single-asset tests have none), the cross-asset Itô recovery is a new mechanism, and the interesting cross-asset signal lives in the worst-conditioned directions. Two gates (M0 engine, M-I cross-Itô) run **before** any hedging logic.
> Scope: **2 or 3 assets**, closed-form-covector payoffs only (basket/spread forwards and Asians), mean-variance plus the existing convex asymmetric penalty. Spread/basket *options*, >3 assets, and arbitrary payoffs stay deferred.

---

## 1. The multi-asset mathematics (detailed)

### 1.1 The path and channels
`M` tradable assets `X¹,…,X^M`. The base path is time-augmented `(t, X¹,…,X^M)`. Then **lead-lag each price channel**, so the enlarged path has channels

`time`, and for each asset `m`: `lead_m`, `lag_m`  →  **`d = 1 + 2M` channels**.

`S` is the (running or terminal) signature of this enlarged path. The **trade letter `a_m`** for asset `m` is its lead channel (the channel that makes `⟨ℓ·a_m, S⟩` the **Itô** integral `∫θ^m dX^m`, exactly as the single-asset trade letter did — validated here by M-I).

### 1.2 Strategy: one covector per asset, each on the *joint* signature
The position in asset `m` at time `t` is

`θ^m_t = ⟨ℓ^m, S_{0,t}⟩`,

a covector `ℓ^m` against the **joint** running signature. The key multi-asset feature is here: `θ^m_t` may depend on the trajectories of **all** assets, not just asset `m` — this is what makes cross-hedging and relative-value possible. The portfolio vector is `(θ^1_t,…,θ^M_t)`.

### 1.3 Trading P&L and the loss covector
Total trading P&L `= Σ_m ∫θ^m dX^m = Σ_m ⟨ℓ^m·a_m, S⟩` (M trade letters). The loss covector for a liability with payoff covector `f` and premium `p₀`:

`𝓛 = f − p₀·∅ − Σ_{m=1}^M ℓ^m·a_m`,   realised loss `L = ⟨𝓛, S⟩`.

`f` is a closed-form covector on the joint path: a **basket forward** `Σ_i w_i X^i_T − K` is level-1 (`(Σ w_i X^i_0 − K)∅ + Σ_i w_i·(price-letter_i)`); a **spread forward** `X^i_T − X^j_T − K` is level-1 with weights `+1, −1`; basket/spread **Asians** are level-2 (the per-asset `(price_i, time)` words, as in single-asset).

### 1.4 Objective and the block-structured dependency matrix
`R(ℓ^1,…,ℓ^M) = E[P(L)] = ⟨P^⧢(𝓛), E[S]⟩`, with `E[S]` the **joint** expected signature.

For mean-variance, stack the per-asset coefficients into one vector indexed by `(w, m)` (word `w` up to strategy depth, asset `m`). Then `R = const − 2 b·ℓ + ℓᵀ A ℓ` with

- `A_{(w,m),(v,n)} = ⟨(w·a_m) ⧢ (v·a_n), E[S]⟩`,
- `b_{(w,m)} = ⟨(f − p₀∅) ⧢ (w·a_m), E[S]⟩`.

`A` has an **`M×M` block structure** indexed by asset pairs:
- **Diagonal blocks** `A^{mm}` — within-asset-`m` dependency (the single-asset `A`, but computed against the joint `E[S]`).
- **Off-diagonal blocks** `A^{mn}` (`m≠n`) — the **cross-asset** dependency, `⟨(w·a_m)⧢(v·a_n), E[S]⟩`. These are nonzero **iff** the joint signature has mixed coordinates involving both assets' channels — i.e. iff the assets are dependent. **The off-diagonal blocks are the entire cross-asset story.**

Solve `(A + λI)ℓ = b`. Independence ⇒ off-diagonal blocks vanish ⇒ `A` block-diagonal ⇒ the solve **decouples** into `M` single-asset hedges. Positive correlation ⇒ off-diagonal blocks nonzero ⇒ `A⁻¹` acquires **negative** cross-asset off-diagonals ⇒ one weight flips short ⇒ a relative-value / pairs position.

### 1.5 Convex asymmetric case
`R(ℓ)` is degree-`q` in the stacked `ℓ`, convex if `P` is convex; minimise by Newton with the block gradient/Hessian (analytic shuffle contractions, as single-asset). Cross-asset **co-skewness / co-kurtosis** enter the higher-degree blocks — the multi-asset analog of the single-asset higher moments.

### 1.6 Truncation–degree coupling and dimensionality
With `d = 1 + 2M` channels, the joint signature has `~dᴺ` coordinates. The objective needs `E[S]` to level `N_sig ≥ q·d_strat`. Concrete budget (the reason for the 2–3 asset cap):

| M | d | strategy coeffs (d_strat=1) | d_strat=1 needs E[S] level (q=2) | coords at that level |
|---|---|---|---|---|
| 2 | 5 | 2·6 = 12 | 2 | ~31 |
| 3 | 7 | 3·8 = 24 | 2 | ~57 |

Push `d_strat=2` only at `M=2` (E[S] to level 4, ~781 coords). At `M=3` hold `d_strat=1`. The binding constraint is the Monte-Carlo estimation of the high-level **cross** coordinates — the noisiest and the most important.

### 1.7 The engine oracle (closed form)
`M` correlated Brownian motions `W¹,…,W^M` (correlation `ρ`), time-augmented, have expected signature `exp_⊗(T·ξ)` with

`ξ = e_time + ½ Σ_{i,j} ρ_{ij} (e_i ⊗ e_j)`   (level-1: only time; level-2: half the correlation matrix).

The symmetric cross coordinate satisfies `E[⟨(i,j),S⟩ + ⟨(j,i),S⟩] = ρ_{ij}·T`, and the antisymmetric (Lévy-area) part `E[⟨(i,j)⟩ − ⟨(j,i)⟩] = 0`. In log-price coordinates for correlated GBMs the generator is `ξ = e_time + Σ_i(r−σ_i²/2)e_i + ½Σ_{ij}σ_iσ_jρ_{ij}(e_i⊗e_j)`; the cross coordinate is `σ_iσ_jρ_{ij}T`.

---

## 2. Test gates (engine and cross-Itô first; then hedging)

Run under the risk-neutral measure (drift `=r`); incompleteness for the risk-profile gate comes from multi-asset Heston or coarse rebalancing. Wire every gate into the inventory. Statistical gates: seed-swept or generous margin. **Mutation-test the new gates** (Section 4).

- **M0 — Joint-signature engine oracle (FIRST, before any hedging code).** `M=2` and `M=3` correlated BMs (and a log-GBM variant). Compare the MC joint expected signature to `exp_⊗(T·ξ)` at all levels; in particular assert the symmetric cross coordinate `= ρ_{ij}T` (per pair) and the antisymmetric cross part `= 0` within `4·SE`. *Why: a sign/ordering error in cross-asset coordinates is invisible to every single-asset test; everything below sits on these.*
- **M-I — Cross-asset Itô recovery (FIRST).** From the joint lead-lag, recover `∫X^i dX^j` (i≠j) and the cross-variation `[X^i,X^j]`, and match the closed form (`[X^i,X^j]_T = ρσ_iσ_j∫X^iX^j dt`; the Itô cross integral differs from Stratonovich by `½[X^i,X^j]`). Per-path to tight tolerance. *Why: test I only validated the same-asset `∫X dX`; the cross integral is a new mechanism.*
- **MH0 — Objective cross-check (BEDROCK).** Multi-asset `R(ℓ) = ⟨P^⧢(𝓛),E[S]⟩` equals a direct-MC `E[P(L)]` with the P&L summed as `Σ_m Σ_k θ^m_k·(X^m_{k+1}−X^m_k)` over **literal** simulated increments — independent of every signature convention. Several random `ℓ`, both penalties, within `4·SE`. *Why: the same independent guard that anchored the single-asset phase; the literal multi-asset P&L cannot be faked by a coordinate bug.*
- **MH1 — Cross-asset replication oracles (complete GBM, r=0).**
  - **Spread forward** `X¹−X²−K` → solve must recover `θ¹=+1, θ²=−1` (constant), `E[L²]≈0`. This is the clean long-short / pairs structure emerging from a derivative.
  - **Basket forward** `Σ w_i X^i − K` → `θ^i = w_i`, `E[L²]≈0`.
  *Why: known exact multi-asset hedges; the analogs of single-asset H1.*
- **MH2 — Level-0 = multivariate regression.** The depth-0 mean-variance hedge equals the vector regression `ℓ⁰ = Σ_{ΔX}⁻¹ Cov(F, ΔX)` (covariance of the payoff with the asset-increment vector). Oracle: a direct-MC estimate of that regression. Within `4·SE`.
- **MH3 — Off-diagonal blocks vanish ⇔ independence (the cross-asset machinery, made structural).** Assemble `A` for independent assets and for correlated assets. Assert: independent ⇒ `‖off-diagonal blocks‖ ≈ 0` (block-diagonal) and the solve equals two independent single-asset solves; correlated ⇒ off-diagonal blocks materially nonzero and the joint solve **differs** from the decoupled one. *Why: directly exhibits the off-diagonal blocks doing their job; confirms the cross-asset code is vacuous exactly when it should be (independence) and active when it should be.*
- **MH4 — Conditioning and the pairs/ridge tension (the subtle gate).** (a) `cond(A)` grows with `M` **and** with correlation (report it; highly correlated assets → near-singular `A`). (b) Ridge stabilises the solve across seeds. (c) **Critically:** the spread-forward `(+1,−1)` of MH1 must still be recovered **after** ridging — assert the ridged solution preserves the relative-value position, not just numerical stability. *Why: the cross-asset signal lives in the near-null-space direction ridge damps; this gate proves the ridge stabilises without erasing the pairs structure.*
- **MH5 — Cross-hedge value (correlated assets).** Hedge a payoff on `X¹` using **only `X²`**: residual `Var(L)` must fall below the unhedged level iff `ρ≠0`, and decrease with `|ρ|`; for independent assets `θ²*≈0` and no reduction. *Why: isolates the pure cross-hedge — using one asset to hedge another's risk — which exists only through the off-diagonal.*
- **MH6 — Asymmetric penalty thins the tail (multi-asset, incomplete).** Under multi-asset Heston (or coarse rebalancing), with correlated assets: the convex-quartic hedge shows strictly lower `CVaR₉₅(L)` and strictly higher `Var(L)` than the mean-variance hedge, now exploiting cross-asset **co-skewness** to reshape the joint tail. Report variance, 95% CVaR, `P[L>τ]` for MV vs asymmetric. *Why: the headline feature, multi-asset version; meaningless without both correlation and incompleteness.*

**Edge cases:** `ℓ=0` → unhedged baseline; `p₀=price` → `E[L]≈0`; `M=2` and `M=3` both exercised; independent and correlated regimes both run for MH3/MH5.

---

## 3. Reporting

Extend the `[HEDGE]` block / inventory with a multi-asset section: M0 cross-coordinate vs `ρT` per pair; M-I cross-Itô error; MH1 recovered portfolio vectors (`(+1,−1)`, basket weights); MH3 off-diagonal-block norm, independent vs correlated; MH4 `cond(A)` by `M` and `ρ`, plus the post-ridge spread-position check; MH6 the MV-vs-asymmetric tail table.

---

## 4. Mutation tests for the new gates (run, don't assume)

Inject and confirm a gate goes red:
- Swap trade letters `a_1 ↔ a_2` → MH0/MH1 must fire (wrong asset gets the P&L).
- Zero the cross-asset coordinates in the joint signature → M0 must fire (cross coordinate ≠ ρT) and MH3/MH5 must fire (cross-hedge vanishes).
- Negate a cross-variation term in the lead-lag → M-I must fire.
- Over-ridge (`λ` large) → MH4(c) must fire (spread position `(+1,−1)` no longer recovered). *This is the gate that proves you haven't quietly ridged away the cross-asset signal.*
- Force `θ²=0` (disable cross-hedging) → MH5 must fire (no variance reduction from the correlated asset).

---

## 5. Non-goals (deferred)

- **Spread/basket options** (`max(·,0)`) and any non-linear multi-asset payoff — need regression and have no closed-form covector; deferred with the arbitrary-payoff path.
- **More than 3 assets** — `dᴺ` and the estimation wall make it the next tractability frontier, not this phase.
- **Non-convex penalties, physical-measure P&L, transaction costs** — as in the single-asset phase.
- **Exogenous signals** (the Sig-Trading "factors") — a natural follow-on (more channels in the path), but out of scope here.

---

## 6. Lingering issues to hold in view

1. **The cross-asset value and the cross-asset fragility are the same directions.** Relative-value / pairs structure lives in the near-null-space of a near-singular `A` (correlated assets = near-redundant instruments). Ridge stabilises but damps it. MH4(c) is the guard; do not let it become soft.
2. **Independence makes the whole extension vacuous** — every cross-asset gate must use correlated assets, and the *replication-structure* gates (MH1) run complete while the *risk-profile* gate (MH6) needs incompleteness. Conflating the two regimes will make a correct result look broken or a broken one look fine.
3. **The estimation wall is now severe**, not benign: the high-level cross coordinates are the noisiest and the most load-bearing. Watch their relative SE (the D4-style check, now on cross coordinates); if it is poor, the cross-hedge is being fit to noise.
4. **The engine and cross-Itô oracles are non-negotiably first.** Build any hedging logic before M0/M-I are green and you are building on an unvalidated joint signature that no single-asset test can vet.