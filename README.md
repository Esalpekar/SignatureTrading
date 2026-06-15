# Signature Trading

Hedging a portfolio of arbitrary, path-dependent derivatives — under a risk
profile *you* choose. A validated Python core (`sigcore`) plus an interactive
web demo, built on the signature methods of Lyons et al. (2019).

**Live demo:** https://esalpekar.github.io/SignatureTrading/

---

## The problem with exotic derivatives

Derivatives often have simple rules. A call option pays `X_T − K`: the asset's
price at maturity, minus a fixed strike. You could explain it on a napkin. But
the moment a contract's rules get more *exotic*, the mathematics needed to
analyze it tends to explode — and each new payoff seems to demand its own
bespoke theory. The study of well-behaved contracts paralyzes the study of
interesting ones.

This framework refuses that fragmentation. It hedges **any** derivative whose
payoff is a continuous function of the asset's price path `X_t`. A call option
reads a single number off the path — its endpoint. A *variance swap* reads
something much richer: how much the price *moved* over the window, regardless of
where it ended up. These two contracts feel nothing alike. But both are
continuous functions of the same path, and that is the only property the method
needs. They are hedged by one machine.

## Beyond mean-variance

What's more, the underlying theory conveniently lets you optimize for an
**arbitrary attitude toward risk**. Much of the portfolio-hedging literature
descends from *mean-variance* hedging, pioneered more than sixty years ago — and
it shows its age. Mean-variance penalizes upside and downside symmetrically: a
surprise gain is treated as exactly as regrettable as an equal surprise loss.
Real traders are not so even-handed.

Here, a trader who fears losses more than they crave gains — or who has some
situational need to shape a particular higher-order feature of the
return distribution — can hedge directly against *that* preference, written as a
loss penalty. The demo takes this preference as an input and shows the resulting
hedge side by side with the mean-variance answer, so you can watch the
difference.

## Signatures, from the ground up

The workhorse is **signature theory**. A signature is a way of turning a whole
path into numbers — many numbers, arranged in levels.

Take the asset's price path and enrich it slightly: add a clock, and a
"lead-lag" copy of the path against itself (a standard trick that makes
*volatility* visible to the integrals below). Now read off the levels:

- **Level 1** records the net change in each channel — where the path ended up.
- **Level 2** records *signed areas* — how pairs of channels co-move over time.
  This captures order and interaction that a net change throws away (it is, for
  instance, how realized variance becomes legible).
- **Level 3** records a still finer interaction, and so on up.

Truncate at some depth and you have a finite feature vector that summarizes the
entire path — not just its endpoints, but its shape through time.

A central property of the signature is that any continuous function of the path
can be approximated, to arbitrary accuracy, by a linear combination of signature
terms. A quantity that depends on the whole path in a nonlinear way can therefore
be computed as a single inner product against the signature. The signature turns
path-dependent functions into linear ones.

## Why this makes hedging tractable

Two consequences follow from that property.

First, a trading strategy is itself a function of the path. The holding in an
asset at time t can depend on everything observed up to t, which makes it a
feedback rule rather than a fixed schedule. Such a strategy is a linear readout
of the running signature S_{0,t}: fix a vector of coefficients ℓ, and the holding
at time t is the inner product ⟨ℓ, S_{0,t}⟩. Reading those coefficients off the
signature as the path develops produces the full sequence of holdings.

Second, the product of two functions that are linear in the signature is again
linear in the signature. The operation that achieves this is the shuffle product,
and it keeps the optimization finite. The terminal loss L is linear in the
signature. The risk penalty P(L), a polynomial in L that sets how much each size
of loss is disliked, is therefore also linear in the signature after its shuffle
expansion. Its expectation reduces to one inner product:

```
E[ P(L) ]  =  ⟨ P^shuffle(ℓ),  E[S] ⟩
```

E[S] is the expected signature, and it is the only place the model of the asset
enters. It is obtained from whatever describes the price dynamics, such as GBM or
Heston, in closed form where available and by Monte Carlo otherwise. The rest of
the calculation does not depend on the model.

Minimizing expected risk over all feedback strategies is, stated directly, a
search over functions. The reduction above replaces it with a finite optimization
over the coefficient vector ℓ against a fixed E[S].

## The demo

The live demo (https://esalpekar.github.io/SignatureTrading/) takes three inputs:

- a model of the underlying: GBM or Heston;
- a contract: forward, Asian, or variance swap;
- a risk profile: the loss polynomial that sets how steeply larger losses are
  penalized.

It returns the optimal hedge, the resulting profit-and-loss distribution, and the
mean-variance hedge for comparison. Adjusting the risk profile changes the shape
of the loss tail. A separate view traces the optimal allocation of a three-asset
basket on a simplex.

## Repository

- `sigcore/`: signatures, models (GBM, Heston), pricing, and hedging.
- `tests/`: acceptance and unit tests that pin the core to analytic ground truth.
- `report.py`: validation harness comparing closed-form and Monte Carlo results.
- `webapp/`: React frontend and FastAPI backend. The live demo reads a
  precomputed static databank (`webapp/build_databank.py`) and needs no server.
- `background/`: the source papers.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # tests
python report.py   # validation report
./webapp/run.sh    # local web app (frontend + backend)
```

## References

- Lyons, Nejad & Pérez Arribas, Nonparametric pricing and hedging of exotic
  derivatives (2019).
- Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework
  with Exogenous Signals.

Both are in `background/`.
