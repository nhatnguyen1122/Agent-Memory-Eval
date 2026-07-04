#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EVAL_DIR}"
source "${SCRIPT_DIR}/provider_profile.sh"

CONDA_ENV="${CONDA_ENV:-mem}"
PROJECT_PREFIX="${PROJECT_PREFIX:-${LLM_PROFILE}-longmemeval}"
BACKENDS="${BACKENDS:-mem0 mem0_graph a_mem memorybank}"

MEMORY_EMBEDDER_MODEL="${MEMORY_EMBEDDER_MODEL:-nvidia/llama-nemotron-embed-1b-v2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RPM="${RPM:-60}"
TOP_K="${TOP_K:-200}"
TOP_K_CUTOFFS="${TOP_K_CUTOFFS:-10,20,50,200}"
QUESTION_TYPES="${QUESTION_TYPES:-}"
PER_TYPE="${PER_TYPE:-}"
ALL_QUESTIONS="${ALL_QUESTIONS-1}"

for backend in ${BACKENDS}; do
  project_name="${PROJECT_PREFIX}-${backend}"
  echo "Running LongMemEval with backend=${backend} project=${project_name} profile=${LLM_PROFILE}"
  all_questions_arg=()
  if [[ "${ALL_QUESTIONS}" == "1" || "${ALL_QUESTIONS}" == "true" || "${ALL_QUESTIONS}" == "yes" ]]; then
    all_questions_arg=(--all-questions)
  fi
  conda run -n "${CONDA_ENV}" python -m benchmarks.longmemeval.run \
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
    ${QUESTION_TYPES:+--question-types "${QUESTION_TYPES}"} \
    ${PER_TYPE:+--per-type "${PER_TYPE}"} \
    "${all_questions_arg[@]}"
done
