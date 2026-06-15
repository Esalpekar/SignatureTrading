# Signature Hedging — Web Demo (v1) Implementation Spec

> Audience: a coding agent building a React + Tailwind front end over the validated `sigcore` library.
> The page has one job: let a visitor pick a *world*, a *contract*, and a *risk attitude*, and **see** the optimal hedge and how the risk attitude reshapes the loss tail. Built for a vaguely-technical viewer (a recruiter) who wants to test the depth; the whitepaper and notebook serve the deeper audience.
> Everything renders from **real `sigcore` computation** — no mocked data, no fabricated curves. The integrity that earned the green test suite carries into the demo.

---

## 0. The one correctness constraint — read before anything else

The risk-profile selector is the centerpiece interaction, and **it does nothing in a complete market.** Under fine-grid GBM every payoff replicates perfectly, so changing the loss polynomial leaves the hedge and the P&L identical — the hero interaction would be dead and look broken when it is the mathematics. Therefore the demo **defaults to an incomplete setting**: Heston (stochastic volatility, the validated simulator) or GBM with deliberately coarse rebalancing. Only in that regime does dragging the risk profile visibly move the loss tail. This is the same fact as test gates H6/MH6; do not build the demo on complete GBM.

---

## 1. Architecture

- **Backend:** a thin FastAPI service wrapping `sigcore`. Endpoints:
  - `POST /paths` → a handful of sample price paths for the model visual.
  - `POST /price` → the fair price `p₀` of the selected derivative.
  - `POST /hedge` → the fitted strategy `ℓ*` plus the holdings path `θ_t` evaluated on a few representative sample paths.
  - `POST /pnl` → the shortfall distribution for **both** the mean-variance hedge and the selected polynomial: running cumulative P&L on sample paths, the terminal-P&L histogram, and the summary numbers (variance, 95% CVaR, `P[loss > τ]`).
  All numbers come from `sigcore`; the endpoint is a wrapper, not a reimplementation.
- **Front end:** React + Tailwind. Charts hand-rolled in SVG or via a lightweight library themed to the monochrome palette (avoid a charting library's default colored theme — it will fight the aesthetic).
- **Static fallback:** if zero-backend hosting is wanted, precompute a grid of `(model, derivative, polynomial)` results to static JSON and have the front end read it (limits interactivity to the precomputed grid). Prefer the backend for genuine interactivity.
- **Latency:** fitting `ℓ*` and simulating the P&L take real time. Show a loading state on parameter change, debounce slider input, keep `n_paths` moderate (enough for a smooth histogram, not so many the UI stalls), and cache by parameter hash.
- **No** accounts, database, persistence, real market data, or models beyond GBM/Heston.

---

## 2. Inputs (three controls, each with a plain-language explanation)

Every control carries a short explanation written from the user's side of the screen (not system internals). The four steps are a genuine pipeline — *world → contract → risk attitude → result* — so a **numbered, sequential layout is justified** (the numbering encodes the real data flow, §5).

**Step 1 — Model of the underlying** (with a visual).
Options: **GBM** and **Heston**; expose their parameters (GBM: `r`, `σ`; Heston: `κ, θ, ξ, ρ, v₀`). Visual: a few sample price paths in monochrome plus the terminal-price distribution, so the visitor sees the world they are in. **Default to Heston** (incompleteness, §0). Explanation: "the assumed behaviour of the asset price over time."

**Step 2 — Derivative** (presets with pre-formed, closed-form covectors — including ones people don't usually think of as derivatives).
All presets are **linear functionals of the signature** (closed-form covectors; the arbitrary-payoff path is deferred). Show each preset's covector `f` rendered in the monospace face, beside a plain "what you're betting on" line. Presets:
- **Forward** — `X_T − K`. "The price at maturity." (Level 1 — the ordinary case.)
- **Asian (average)** — average price `− K`. "The average price over the life, not just the end." (Level 2.)
- **Weighted average** — a time-tilted average. "An average that weights later dates more." (Level 2.)
- **Realised-variance swap** — pays realised variance `− K` (the quadratic variation, via the lead-lag). "A bet on *how much* the price moved, regardless of where it ended."
- *(2-asset mode)* **Spread forward** — `X¹_T − X²_T − K`. "The gap between two assets." (Hedges to a long-short / pairs position.)
- *(2-asset mode)* **Covariance swap** — pays realised covariance `[X¹,X²]`. "A bet on how much two assets moved *together*."
- *(2-asset mode)* **Lévy-area claim** — the signed area between two assets' paths. "A bet on *which asset moved first*."
The realised-variance, covariance, and Lévy-area claims are the showcase: they are things rarely framed as tradeable contracts, yet in this framework they are the **same kind of object** as a vanilla forward — just a different signature coordinate. The covector display makes that visible and is the depth-signal for the technical viewer. Explanation on the control: "the contract you are hedging — what you owe at maturity."

**Step 3 — Loss polynomial** (templates + a live graph).
Templates: **Mean-variance** (`P(x)=x²`) and **Downside-averse** (the convex quartic `P(x)=x²+γx³+δx⁴`) with `γ, δ` adjustable; optionally a **tail-averse** preset (heavier even weighting). **Enforce convexity** (`δ ≥ 3γ²/8`) in the control — keep the sliders inside the convex region and indicate the boundary, since a non-convex penalty leaves the validated regime. Show a **live plot of `P(x)`** over a symmetric range so the asymmetry is visible (losses to the right penalised more steeply than equal gains to the left). Explanation: "how you weigh losses of different sizes. The cubic term tilts the penalty toward losses — you fear a big loss more than you value an equal gain."

---

## 3. Outputs (the visuals)

**Optimal holdings path `θ_t`.** The position over time, evaluated on a representative sample path (single asset: one line; 2-asset: two, showing the long-short structure when it appears). Show that it *reacts to the path* — it is a feedback rule, not a fixed schedule — ideally by overlaying the holdings for two or three different sample paths.

**P&L — the centerpiece.** Two linked views: the **running cumulative P&L** (a fan of sample paths with a band) and the **terminal-P&L histogram**. The histogram is the thesis of the whole project: overlay the distribution under **mean-variance** against the distribution under the **selected risk profile**, so the visitor sees the **left (loss) tail thinned and the variance widened** as they move to a downside-averse profile. Beside it, the honest numbers: `p₀`, and variance / 95% CVaR / `P[loss > τ]` for the selected profile versus mean-variance.

**Monochrome series.** Because the palette is grayscale, the two distributions (and any overlaid series) must be distinguished by **texture and weight — solid vs hatched fill, line weight, dash pattern — never by colour.** This is a real constraint to design around, not an afterthought.

---

## 4. The signature interaction (the hero)

The single memorable moment, and what the page should be built around: **dragging the risk-profile sliders and watching the terminal-P&L histogram's loss tail move in real time** — the left tail retreating, the variance broadening, the CVaR number dropping — under an incomplete model. This *is* the project's thesis made tangible: a personalised risk attitude visibly trading variance for a thinner tail. Make it the centre of the layout and animate the transition smoothly; do not bury it below a stack of controls, and do not scatter unrelated motion elsewhere.

---

## 5. Design direction (grounded in the subject)

The subject is a rigorous quantitative-finance instrument with an academic-paper lineage — precise, restrained, data-forward. The aesthetic should read as a serious tool, not a SaaS landing page. A strong starting token system (refine, but derive every decision from the subject, not from a generic minimalist default):

- **Palette — grayscale only.** `ink #141414` (text, lines), `paper #FCFCFC` (background), `slate #6B6B6B` (secondary text), `grid #B8B8B8` (axes, hairlines), `divider #E5E5E5`, `fill #ECECEC` (chart fills). No colour accent — let weight, contrast, and texture carry hierarchy. (At most one restrained ink for a single loss-tail emphasis, if needed; default to pure grayscale, per the brief.)
- **Typography — a deliberate trio.** A clean neutral **grotesque** for UI, labels, and display, used with restraint; a quiet **text face** for the explanatory prose (a serif here grounds the academic-rigour identity, if you want the paper feel); and a **monospace** for all data, numbers, and covectors. The monospace is the subject-driven choice — the math is first-class, so it gets its own face. Set a clear type scale with intentional weights.
- **Layout — a numbered pipeline.** Steps 1–3 (model, contract, risk attitude) as compact, sequentially numbered controls, feeding a large central **Result** panel (holdings + P&L). The numbering is honest here because the content *is* a sequence; do not number things that are not.
- **Structure encodes content.** Hairline dividers, restrained labels, the covector shown as data — every structural device should mean something (the covector display literally shows the contract's mathematical identity), not decorate.
- **Motion — one orchestrated moment.** The tail-reshaping transition is where motion serves the subject; keep everything else still. Extra animation reads as generated.
- **Copy — from the user's side, active, plain.** "The contract you're hedging," not "payoff covector input." Each label does one job. Include a small **"Assumptions"** note for the technical viewer — model-based, simulated data, the favourable stationary regime, arbitrary payoffs out of scope — linking to the whitepaper. Honesty over polish.
- **Quality floor (unannounced):** responsive down to mobile, visible keyboard focus, `prefers-reduced-motion` respected.

---

## 6. Honesty and integrity (carry forward from the library)

- Every curve and number comes from a real `sigcore` call. No placeholder data, no hand-drawn "representative" shapes.
- The **Assumptions** note states the scope plainly: generated data from a chosen model, the stationary regime where the method is validated, and the deferred pieces (arbitrary payoffs, real-market robustness). A QR reading it should find no overclaim.
- If a visitor selects a **complete** configuration (fine-grid GBM), the risk profile will not change the hedge. Either gate that configuration out of the risk-profile view, or surface a one-line note explaining why nothing moves — never let a correct result look like a bug.

---

## 7. Non-goals

No arbitrary-payoff or free-form input (presets only). No real market data. No accounts, database, or saved sessions. No models beyond GBM and Heston. Keep the 2-asset mode as a clearly-marked secondary view; the single-asset path is the primary, cleanest demo.

---

## 8. Build gotchas

- **Default Heston (or coarse rebalancing)** so the hero interaction is alive. This is the single most important non-aesthetic decision.
- **Enforce convexity** (`δ ≥ 3γ²/8`) in the polynomial sliders.
- **Distinguish chart series by texture/weight, not colour** — the monochrome brief makes this load-bearing.
- **Debounce and cache** the backend calls; fitting a hedge and simulating P&L is not instant.
- Render the **covector** for each derivative in the monospace face — it is the depth-signal that rewards a technical viewer for looking closely.