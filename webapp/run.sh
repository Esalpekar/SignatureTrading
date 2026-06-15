#!/usr/bin/env bash
# Start both servers for the demo. Backend MUST run from the project venv
# (where sigcore + iisignature + fastapi are installed), not system python.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then echo "expected project venv at .venv (see README)"; exit 1; fi
# Use the venv interpreter directly (robust if the project dir was renamed,
# which breaks the absolute paths baked into .venv/bin/activate).
PY="$ROOT/.venv/bin/python"
"$PY" -c "import fastapi" 2>/dev/null || "$PY" -m pip install -r webapp/requirements.txt

echo "starting backend on :8000 (sigcore + fastapi, from .venv) ..."
PYTHONPATH=webapp "$PY" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --log-level warning &
BACKEND=$!
trap "kill $BACKEND 2>/dev/null" EXIT

# wait for backend health
until curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; do sleep 0.5; done
echo "backend ready. starting frontend dev server ..."

cd webapp/frontend
[ -d node_modules ] || npm install
npm run dev
