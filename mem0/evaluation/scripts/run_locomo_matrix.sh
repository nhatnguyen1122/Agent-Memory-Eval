#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EVAL_DIR}"

CONDA_ENV="${CONDA_ENV:-mem}"
NVIDIA_BASE_URL="${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}"
NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
PROJECT_PREFIX="${PROJECT_PREFIX:-nim-locomo}"
BACKENDS="${BACKENDS:-mem0 mem0_graph a_mem memorybank}"

ANSWERER_MODEL="${ANSWERER_MODEL:-meta/llama-3.1-70b-instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-meta/llama-3.1-70b-instruct}"
MEMORY_MODEL="${MEMORY_MODEL:-meta/llama-3.1-70b-instruct}"
MEMORY_EMBEDDER_MODEL="${MEMORY_EMBEDDER_MODEL:-nvidia/nv-embedqa-e5-v5}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RPM="${RPM:-60}"
TOP_K="${TOP_K:-200}"
TOP_K_CUTOFFS="${TOP_K_CUTOFFS:-10,20,50,200}"
CONVERSATIONS="${CONVERSATIONS:-0,1,2,3,4,5,6,7,8,9}"
CATEGORIES="${CATEGORIES:-1,2,3,4}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"

if [[ -z "${NVIDIA_API_KEY}" ]]; then
  echo "NVIDIA_API_KEY is required."
  exit 1
fi

for backend in ${BACKENDS}; do
  project_name="${PROJECT_PREFIX}-${backend}"
  echo "Running LOCOMO with backend=${backend} project=${project_name}"
  conda run -n "${CONDA_ENV}" python -m benchmarks.locomo.run \
    --project-name "${project_name}" \
    --memory-backend "${backend}" \
    --provider openai \
    --base-url "${NVIDIA_BASE_URL}" \
    --api-key "${NVIDIA_API_KEY}" \
    --answerer-model "${ANSWERER_MODEL}" \
    --judge-model "${JUDGE_MODEL}" \
    --memory-api-key "${NVIDIA_API_KEY}" \
    --memory-base-url "${NVIDIA_BASE_URL}" \
    --memory-model "${MEMORY_MODEL}" \
    --memory-embedder-model "${MEMORY_EMBEDDER_MODEL}" \
    --max-workers "${MAX_WORKERS}" \
    --rpm "${RPM}" \
    --top-k "${TOP_K}" \
    --top-k-cutoffs "${TOP_K_CUTOFFS}" \
    --conversations "${CONVERSATIONS}" \
    --categories "${CATEGORIES}" \
    ${MAX_QUESTIONS:+--max-questions "${MAX_QUESTIONS}"}
done
