#!/usr/bin/env bash
set -euo pipefail

# OpenAI-compatible chat provider profile.
#
# Usage:
#   LLM_PROFILE=gptoss GPTOSS_API_KEY=... source scripts/provider_profile.sh
#   LLM_PROFILE=qwen QWEN_API_KEY=... source scripts/provider_profile.sh
#   LLM_PROFILE=qwen3 QWEN3_API_KEY=... source scripts/provider_profile.sh
#   LLM_PROFILE=nvidia NVIDIA_API_KEY=... source scripts/provider_profile.sh
#
# Memory embeddings stay separate because GPT-OSS/Qwen are chat models, not
# embedding models. Use NIM embeddings by default:
#   MEMORY_EMBEDDER_API_KEY="${NVIDIA_API_KEY}"
#   MEMORY_EMBEDDER_BASE_URL="https://integrate.api.nvidia.com/v1"

LLM_PROFILE="${LLM_PROFILE:-gptoss}"

case "${LLM_PROFILE}" in
  gptoss)
    export LLM_BASE_URL="${LLM_BASE_URL:-https://stream-netmind.viettel.vn/gateway}"
    export LLM_API_KEY="${LLM_API_KEY:-${GPTOSS_API_KEY:-${NETMIND_API_KEY:-}}}"
    export ANSWERER_MODEL="${ANSWERER_MODEL:-openai/gpt-oss-120b:netmind}"
    export JUDGE_MODEL="${JUDGE_MODEL:-${ANSWERER_MODEL}}"
    export MEMORY_MODEL="${MEMORY_MODEL:-${ANSWERER_MODEL}}"
    ;;
  qwen)
    export LLM_BASE_URL="${LLM_BASE_URL:-https://stream-netmind.viettel.vn/gateway/v1}"
    export LLM_API_KEY="${LLM_API_KEY:-${QWEN_API_KEY:-${NETMIND_API_KEY:-}}}"
    export ANSWERER_MODEL="${ANSWERER_MODEL:-Qwen/Qwen3.5-35B-A3B-FP8}"
    export JUDGE_MODEL="${JUDGE_MODEL:-${ANSWERER_MODEL}}"
    export MEMORY_MODEL="${MEMORY_MODEL:-${ANSWERER_MODEL}}"
    export OPENAI_EXTRA_BODY_ENABLE_THINKING="${OPENAI_EXTRA_BODY_ENABLE_THINKING:-false}"
    ;;
  qwen3)
    export LLM_BASE_URL="${LLM_BASE_URL:-https://stream-netmind.viettel.vn/gateway/v1}"
    export LLM_API_KEY="${LLM_API_KEY:-${QWEN3_API_KEY:-${QWEN_API_KEY:-${NETMIND_API_KEY:-}}}}"
    export ANSWERER_MODEL="${ANSWERER_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
    export JUDGE_MODEL="${JUDGE_MODEL:-${ANSWERER_MODEL}}"
    export MEMORY_MODEL="${MEMORY_MODEL:-${ANSWERER_MODEL}}"
    export OPENAI_EXTRA_BODY_ENABLE_THINKING="${OPENAI_EXTRA_BODY_ENABLE_THINKING:-false}"
    ;;
  nvidia)
    export LLM_BASE_URL="${LLM_BASE_URL:-${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}}"
    export LLM_API_KEY="${LLM_API_KEY:-${NVIDIA_API_KEY:-}}"
    export ANSWERER_MODEL="${ANSWERER_MODEL:-meta/llama-3.1-8b-instruct}"
    export JUDGE_MODEL="${JUDGE_MODEL:-${ANSWERER_MODEL}}"
    export MEMORY_MODEL="${MEMORY_MODEL:-${ANSWERER_MODEL}}"
    ;;
  *)
    echo "Unknown LLM_PROFILE=${LLM_PROFILE}. Use: gptoss, qwen, qwen3, nvidia." >&2
    exit 1
    ;;
esac

if [[ -z "${LLM_API_KEY}" ]]; then
  echo "Missing chat API key for LLM_PROFILE=${LLM_PROFILE}." >&2
  echo "Set LLM_API_KEY or the profile-specific key env var." >&2
  exit 1
fi

export MEMORY_EMBEDDER_MODEL="${MEMORY_EMBEDDER_MODEL:-nvidia/llama-nemotron-embed-1b-v2}"
export MEMORYBANK_EMBEDDING_MODEL="${MEMORYBANK_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
export MEMORY_EMBEDDER_BASE_URL="${MEMORY_EMBEDDER_BASE_URL:-${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}}"
export MEMORY_EMBEDDER_API_KEY="${MEMORY_EMBEDDER_API_KEY:-${NVIDIA_API_KEY:-}}"
export MEMORY_ADD_EMBEDDING_TYPE="${MEMORY_ADD_EMBEDDING_TYPE:-passage}"
export MEMORY_UPDATE_EMBEDDING_TYPE="${MEMORY_UPDATE_EMBEDDING_TYPE:-passage}"
export MEMORY_SEARCH_EMBEDDING_TYPE="${MEMORY_SEARCH_EMBEDDING_TYPE:-query}"

if [[ -z "${MEMORY_EMBEDDER_API_KEY}" ]]; then
  echo "Missing embedding API key. Set NVIDIA_API_KEY or MEMORY_EMBEDDER_API_KEY." >&2
  exit 1
fi
