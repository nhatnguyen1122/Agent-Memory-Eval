#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EVAL_DIR}"
source "${SCRIPT_DIR}/provider_profile.sh"

CONDA_ENV="${CONDA_ENV:-mem}"
PROJECT_PREFIX="${PROJECT_PREFIX:-${LLM_PROFILE}-locomo}"
BACKENDS="${BACKENDS:-mem0 mem0_graph a_mem memorybank}"

MEMORY_EMBEDDER_MODEL="${MEMORY_EMBEDDER_MODEL:-nvidia/llama-nemotron-embed-1b-v2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RPM="${RPM:-60}"
TOP_K="${TOP_K:-200}"
TOP_K_CUTOFFS="${TOP_K_CUTOFFS:-10,20,50,200}"
CONVERSATIONS="${CONVERSATIONS:-0,1,2,3,4,5,6,7,8,9}"
CATEGORIES="${CATEGORIES:-1,2,3,4}"
MAX_QUESTIONS="${MAX_QUESTIONS:-}"
QUESTION_OFFSET="${QUESTION_OFFSET:-0}"

for backend in ${BACKENDS}; do
  project_name="${PROJECT_PREFIX}-${backend}"
  echo "Running LOCOMO with backend=${backend} project=${project_name} profile=${LLM_PROFILE}"
  conda run -n "${CONDA_ENV}" python -m benchmarks.locomo.run \
    --project-name "${project_name}" \
    --memory-backend "${backend}" \
    --provider openai \
    --base-url "${LLM_BASE_URL}" \
    --api-key "${LLM_API_KEY}" \
    --answerer-model "${ANSWERER_MODEL}" \
    --judge-model "${JUDGE_MODEL}" \
    --memory-api-key "${LLM_API_KEY}" \
    --memory-base-url "${LLM_BASE_URL}" \
    --memory-model "${MEMORY_MODEL}" \
    --memory-embedder-api-key "${MEMORY_EMBEDDER_API_KEY}" \
    --memory-embedder-base-url "${MEMORY_EMBEDDER_BASE_URL}" \
    --memory-embedder-model "${MEMORY_EMBEDDER_MODEL}" \
    --max-workers "${MAX_WORKERS}" \
    --rpm "${RPM}" \
    --top-k "${TOP_K}" \
    --top-k-cutoffs "${TOP_K_CUTOFFS}" \
    --conversations "${CONVERSATIONS}" \
    --categories "${CATEGORIES}" \
    --question-offset "${QUESTION_OFFSET}" \
    ${MAX_QUESTIONS:+--max-questions "${MAX_QUESTIONS}"}
done
