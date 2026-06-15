# Signature Pricing Core — v0 Validation Harness

This harness pins the foundation `simulate → embed → expected signature → price`
to analytic ground truth: it computes an expected signature two independent ways
(Monte Carlo vs Fawcett's closed form) and prices a forward, checking it against
the textbook value `X0 − K·e^{−rT}`. When all four acceptance tests pass, the
core is trusted and the hedging phase may begin.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # iisignature builds against numpy; if it
                                         # fails, run: pip install numpy setuptools wheel
                                         # then: pip install --no-build-isolation iisignature
python report.py                         # motivated report; exits non-zero on failure
pytest -q                                # the four acceptance tests (T1–T4)
```

Checks: **T1** signature engine vs Fawcett, **T2** lead-lag area = ½·quadratic
variation, **T3** Monte-Carlo convergence (slope ≈ −0.5), **T4** forward price
vs closed form.
