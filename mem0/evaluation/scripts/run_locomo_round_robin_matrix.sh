#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Round-robin LoCoMo scheduler.
#
# Category mapping:
#   1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop
#
# Default order follows the requested pattern:
#   multi-hop -> single-hop -> open-domain -> temporal -> next chunk
#
# CHUNK_SIZE=1 means each subprocess handles one question slot for one
# conversation/category. Increase CHUNK_SIZE for larger portions.

export PROJECT_PREFIX="${PROJECT_PREFIX:-${LLM_PROFILE:-gptoss}-locomo-roundrobin}"
export BACKENDS="${BACKENDS:-mem0 mem0_graph a_mem memorybank}"
export CONVERSATIONS="${CONVERSATIONS:-0,1,2,3,4,5,6,7,8,9}"
export CATEGORY_ORDER="${CATEGORY_ORDER:-1 4 3 2}"
export CHUNK_SIZE="${CHUNK_SIZE:-1}"
export LOCOMO_ROUNDS="${LOCOMO_ROUNDS:-100}"

export MAX_WORKERS="${MAX_WORKERS:-1}"
export RPM="${RPM:-20}"
export TOP_K="${TOP_K:-50}"
export TOP_K_CUTOFFS="${TOP_K_CUTOFFS:-10,20,50}"
export MAX_QUESTIONS="${CHUNK_SIZE}"

IFS=',' read -r -a convs <<< "${CONVERSATIONS}"

for ((round=0; round<LOCOMO_ROUNDS; round++)); do
  offset=$((round * CHUNK_SIZE))
  for category in ${CATEGORY_ORDER}; do
    for conv in "${convs[@]}"; do
      export CATEGORIES="${category}"
      export CONVERSATIONS="${conv}"
      export QUESTION_OFFSET="${offset}"

      echo "Round ${round} | category=${category} | conv=${conv} | offset=${offset} | chunk=${CHUNK_SIZE}"
      bash "${SCRIPT_DIR}/run_locomo_matrix.sh"
    done
  done
done
