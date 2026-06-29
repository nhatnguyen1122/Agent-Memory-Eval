#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_PREFIX="${PROJECT_PREFIX:-nim-longmemeval-small}"
export BACKENDS="${BACKENDS:-mem0 mem0_graph a_mem memorybank}"

# Small LongMemEval slice:
# - one multi-session type
# - one single-session type
# - 5 questions per type
export QUESTION_TYPES="${QUESTION_TYPES:-multi-session,single-session-user}"
export PER_TYPE="${PER_TYPE:-5}"
export ALL_QUESTIONS="${ALL_QUESTIONS:-}"

# Safer defaults for Mem0-backed remote runs
export MAX_WORKERS="${MAX_WORKERS:-1}"
export RPM="${RPM:-30}"

# Smaller default models for subset runs; override if needed
export ANSWERER_MODEL="${ANSWERER_MODEL:-meta/llama-3.1-8b-instruct}"
export JUDGE_MODEL="${JUDGE_MODEL:-meta/llama-3.1-8b-instruct}"
export MEMORY_MODEL="${MEMORY_MODEL:-meta/llama-3.1-8b-instruct}"
export MEMORY_EMBEDDER_MODEL="${MEMORY_EMBEDDER_MODEL:-baai/bge-m3}"

# Smaller retrieval surface for quicker debugging
export TOP_K="${TOP_K:-50}"
export TOP_K_CUTOFFS="${TOP_K_CUTOFFS:-10,20,50}"

bash "${SCRIPT_DIR}/run_longmemeval_matrix.sh"
