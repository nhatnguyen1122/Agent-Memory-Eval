#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP="${EVAL_DIR}/demo/mem0_chat_demo.py"
HOST="${STREAMLIT_HOST:-127.0.0.1}"
PORT="${STREAMLIT_PORT:-8501}"

if [[ ! -f "${APP}" ]]; then
  echo "Demo app not found: ${APP}" >&2
  exit 1
fi

cat <<EOF
Starting Mem0 Streamlit demo.

Live chat requires:
  GPTOSS_API_KEY or LLM_API_KEY
  NVIDIA_API_KEY or MEMORY_EMBEDDER_API_KEY

The LoCoMo comparison showcase can run without API keys.

URL:
  http://${HOST}:${PORT}

EOF

if [[ -n "${CONDA_ENV:-}" ]]; then
  conda run -n "${CONDA_ENV}" python -m streamlit run "${APP}" \
    --server.address "${HOST}" \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
else
  python -m streamlit run "${APP}" \
    --server.address "${HOST}" \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
fi
