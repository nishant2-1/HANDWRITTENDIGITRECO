#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Installing dependencies"
python -m pip install -r requirements.txt

echo "[2/4] Running tests"
python -m pytest tests -q

echo "[3/4] Evaluating model"
python -m src.evaluate

echo "[4/4] Starting API and Streamlit"
echo "API docs: http://127.0.0.1:8000/docs"
echo "Streamlit: http://127.0.0.1:8501"

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

python -m streamlit run app/streamlit_app.py
