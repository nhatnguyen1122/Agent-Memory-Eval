# Multi-backend benchmark setup

This setup runs the same `memory-benchmarks` harness against four memory backends:

- `mem0` — current Mem0 with graph/entity linking disabled
- `mem0_graph` — current Mem0 with graph/entity linking enabled
- `a_mem`
- `memorybank`

## Fairness rule

Fairness is enforced by using the same benchmark runner, same dataset, same answerer model, same judge model, same `top_k`, and the same question subsets for every backend. Only the memory backend changes.

## Environment

Recommended conda env:

```bash
bash mem0/evaluation/scripts/setup_conda_env.sh
```

Optional overrides:

```bash
ENV_NAME=memory_eval312 PYTHON_VERSION=3.12 bash mem0/evaluation/scripts/setup_conda_env.sh
```

## Provider Profiles

The scripts support OpenAI-compatible chat providers through `LLM_PROFILE`.
API keys are read from environment variables; do not commit them.

```bash
# GPT-OSS via NetMind/Viettel gateway
export LLM_PROFILE=gptoss
export GPTOSS_API_KEY=...

# Qwen
export LLM_PROFILE=qwen
export QWEN_API_KEY=...

# Qwen3
export LLM_PROFILE=qwen3
export QWEN3_API_KEY=...

# NVIDIA chat
export LLM_PROFILE=nvidia
export NVIDIA_API_KEY=...
```

Mem0 still needs an embedding model. By default the scripts use NVIDIA NIM
embeddings because the NetMind chat models are not embedding models:

```bash
export NVIDIA_API_KEY=...
export MEMORY_EMBEDDER_MODEL=nvidia/llama-nemotron-embed-1b-v2
```

For Qwen/Qwen3, the scripts set `enable_thinking=false` for OpenAI-compatible
chat requests.

## NVIDIA NIM run path

Export the API key and optional model overrides:

```bash
export NVIDIA_API_KEY=...
export NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Example defaults. Override if your NIM account uses different model ids.
export ANSWERER_MODEL=meta/llama-3.1-70b-instruct
export JUDGE_MODEL=meta/llama-3.1-70b-instruct
export MEMORY_MODEL=meta/llama-3.1-70b-instruct
export MEMORY_EMBEDDER_MODEL=nvidia/llama-nemotron-embed-1b-v2
```

Run the full matrix:

```bash
bash mem0/evaluation/scripts/run_locomo_matrix.sh
bash mem0/evaluation/scripts/run_longmemeval_matrix.sh
bash mem0/evaluation/scripts/run_all_matrix.sh
```

Run the smaller subset matrix:

```bash
bash mem0/evaluation/scripts/run_locomo_multihop_small_matrix.sh
bash mem0/evaluation/scripts/run_locomo_singlehop_small_matrix.sh
bash mem0/evaluation/scripts/run_locomo_small_matrix.sh
bash mem0/evaluation/scripts/run_longmemeval_small_matrix.sh
bash mem0/evaluation/scripts/run_small_matrix.sh
```

Run LoCoMo incrementally in round-robin chunks:

```bash
LLM_PROFILE=gptoss \
GPTOSS_API_KEY=... \
NVIDIA_API_KEY=... \
CHUNK_SIZE=1 \
LOCOMO_ROUNDS=100 \
bash mem0/evaluation/scripts/run_locomo_round_robin_matrix.sh
```

Default category order is `1 4 3 2`: multi-hop, single-hop, open-domain,
temporal. Override with `CATEGORY_ORDER="1 2 3 4"` if needed.

Small subset defaults:

- LoCoMo multi-hop: conversation `0`, category `1`, `MAX_QUESTIONS=20`
- LoCoMo single-hop: conversation `0`, category `4`, `MAX_QUESTIONS=20`
- `run_locomo_small_matrix.sh` runs both LoCoMo category-specific jobs
- LongMemEval: question types `multi-session,single-session-user`, `PER_TYPE=5`
- All four backends: `mem0 mem0_graph a_mem memorybank`
- Safer runtime defaults: `MAX_WORKERS=1`, `RPM=30`
- Smaller default LLMs: `meta/llama-3.1-8b-instruct`

## Notes

- `mem0` and `mem0_graph` are the current-repo conditions, not paper-era reproductions.
- `memorybank` here is a lightweight benchmark adapter around MemoryBank-style dated dense retrieval, because the original repo does not provide a reusable LoCoMo / LongMemEval harness.
- `a_mem` uses the patched OpenAI-compatible `base_url` path and persists its benchmark state under `.benchmark_state/`.
- `memory-benchmarks` answerer/judge still use the same OpenAI-compatible endpoint for all backends when `--provider openai --base-url ... --api-key ...` is passed.
- The first real run of `a_mem` or `memorybank` will download sentence-transformer weights if they are not already cached locally.
- LongMemEval exact source-session retrieval metrics are available for new runs only. The harness now stores `source_session_id` provenance in each retrieved memory. Compute `R@5` / `hit@5` with:

```bash
python3 mem0/evaluation/scripts/score_longmemeval_retrieval.py \
  results/longmemeval/longmemeval_results_*.json \
  --cutoffs 5 \
  --output results/longmemeval/longmemeval_retrieval_r5.csv
```

If `missing_source_ids` is nonzero, that row is not an exact retrieval measurement; rerun LongMemEval with the patched harness.
