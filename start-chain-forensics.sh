#!/usr/bin/env bash
set -euo pipefail
CHAIN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CHAIN_PROJECT_ROOT"
# Optional activation script for an existing local runtime.
if [[ -n "${CHAIN_ACTIVATE_SCRIPT:-}" ]]; then
  source "$CHAIN_ACTIVATE_SCRIPT"
elif [[ -x "$CHAIN_PROJECT_ROOT/.venv/bin/python" ]] && "$CHAIN_PROJECT_ROOT/.venv/bin/python" -c 'import fastapi, sklearn, pandas' 2>/dev/null; then
  source "$CHAIN_PROJECT_ROOT/.venv/bin/activate"
elif [[ -f /home/doom/Documents/scripts/activate.sh ]]; then
  source /home/doom/Documents/scripts/activate.sh
fi
command -v python >/dev/null || { echo "Python is required. Follow README.md setup."; exit 1; }
command -v node >/dev/null || { echo "Node.js is required. Follow README.md setup."; exit 1; }
[[ -f frontend/node_modules/vite/bin/vite.js ]] || { echo "Run npm --prefix frontend ci first."; exit 1; }
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python - <<'PY'
import socket
for port in (8000, 5173):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port)) == 0:
            raise SystemExit(f'Port {port} is already serving. Open http://127.0.0.1:5173 if the app is already running, or stop the existing server first.')
PY
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
CHAIN_BACKEND_PID=$!
(cd frontend && exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort) &
CHAIN_FRONTEND_PID=$!
trap 'kill "$CHAIN_BACKEND_PID" "$CHAIN_FRONTEND_PID" 2>/dev/null || true; wait 2>/dev/null || true' EXIT INT TERM
printf '\nChain Forensics: http://127.0.0.1:5173\nPress Ctrl+C here to stop both servers.\n'
wait -n "$CHAIN_BACKEND_PID" "$CHAIN_FRONTEND_PID"
