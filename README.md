# Signature Trading

Signature-based pricing and hedging of exotic derivatives under a personalized
risk profile. A validated Python core (`sigcore`) plus an interactive web demo.

**Live demo:** https://esalpekar.github.io/SignatureTrading/

## Layout

- `sigcore/` — the library: signatures, models (GBM, Heston), pricing, hedging.
- `tests/` — acceptance and unit tests.
- `report.py` — validation harness (analytic ground-truth checks).
- `webapp/` — React frontend + FastAPI backend; the live demo runs off a
  precomputed static databank (`webapp/build_databank.py`).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # tests
python report.py   # validation report
./webapp/run.sh    # local web app (frontend + backend)
```
