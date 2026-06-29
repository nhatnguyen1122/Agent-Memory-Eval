#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EVAL_DIR}"

CONDA_ENV="${CONDA_ENV:-memory_eval312}"
NVIDIA_BASE_URL="${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}"
NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
PROJECT_PREFIX="${PROJECT_PREFIX:-nim-locomo}"
BACKENDS="${BACKENDS:-mem0 mem0_graph a_mem memorybank}"

ANSWERER_MODEL="${ANSWERER_MODEL:-meta/llama-3.1-70b-instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-meta/llama-3.1-70b-instruct}"
MEMORY_MODEL="${MEMORY_MODEL:-meta/llama-3.1-70b-instruct}"
MEMORY_EMBEDDER_MODEL="${MEMORY_EMBEDDER_MODEL:-baai/bge-m3}"

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
    --memory-embedder-model "${MEMORY_EMBEDDER_MODEL}"
done
