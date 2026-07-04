#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_PREFIX="${PROJECT_PREFIX:-nim-locomo-small-singlehop}"
export BACKENDS="${BACKENDS:-mem0 mem0_graph a_mem memorybank}"

# LoCoMo category 4: single-hop.
export CONVERSATIONS="${CONVERSATIONS:-0}"
export CATEGORIES="4"
export MAX_QUESTIONS="${MAX_QUESTIONS:-20}"

# Safer defaults for Mem0-backed remote runs.
export MAX_WORKERS="${MAX_WORKERS:-1}"
export RPM="${RPM:-30}"

# Smaller default models only for NVIDIA profile; provider_profile.sh sets other profiles.
if [[ "${LLM_PROFILE:-gptoss}" == "nvidia" ]]; then
  export ANSWERER_MODEL="${ANSWERER_MODEL:-meta/llama-3.1-8b-instruct}"
  export JUDGE_MODEL="${JUDGE_MODEL:-meta/llama-3.1-8b-instruct}"
  export MEMORY_MODEL="${MEMORY_MODEL:-meta/llama-3.1-8b-instruct}"
fi
export MEMORY_EMBEDDER_MODEL="${MEMORY_EMBEDDER_MODEL:-nvidia/llama-nemotron-embed-1b-v2}"

export TOP_K="${TOP_K:-50}"
export TOP_K_CUTOFFS="${TOP_K_CUTOFFS:-10,20,50}"

bash "${SCRIPT_DIR}/run_locomo_matrix.sh"
