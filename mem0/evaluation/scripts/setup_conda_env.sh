#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../.." && pwd)"

ENV_NAME="${ENV_NAME:-mem}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Using existing conda env: ${ENV_NAME}"
else
  echo "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

echo "Installing benchmark dependencies"
python -m pip install --upgrade pip
python -m pip install -r "${EVAL_DIR}/requirements-multibackend.txt"

echo "Installing editable packages"
python -m pip install -e "${REPO_ROOT}/mem0"
python -m pip install -e "${REPO_ROOT}/A-mem"

echo "Installing spaCy English model"
python -m spacy download en_core_web_sm

echo "Verifying key imports"
python - <<'PY'
import chromadb
import mem0
import openai
import sentence_transformers
import spacy
import agentic_memory
print("Environment ready")
PY

cat <<EOF

Setup complete.

Activate the environment:
  conda activate ${ENV_NAME}

Run the benchmark matrices:
  bash ${EVAL_DIR}/scripts/run_locomo_matrix.sh
  bash ${EVAL_DIR}/scripts/run_longmemeval_matrix.sh

Or run both:
  bash ${EVAL_DIR}/scripts/run_all_matrix.sh
EOF
