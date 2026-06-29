# Experiment Readiness: Mem0, A-mem, MemoryBank on LoCoMo and LongMemEval

## Requested conditions

1. `Mem0-normal`
2. `Mem0-graph`
3. `A-mem`
4. `MemoryBank`

## Bottom line

- `Mem0` has an existing LoCoMo and LongMemEval harness in `mem0/evaluation`.
- `A-mem` does not have a LoCoMo or LongMemEval harness in this repo.
- `MemoryBank-SiliconFriend` does not have a LoCoMo or LongMemEval harness in this repo.
- NVIDIA NIM is the strongest provider candidate because the Mem0 benchmark LLM client already supports OpenAI-compatible `base_url`.
- Mistral is usable only after more provider-specific validation or a dedicated SDK path.

## Evidence by system

### Mem0

- Benchmark repo is present at `mem0/evaluation`.
- LoCoMo runner: `mem0/evaluation/benchmarks/locomo/run.py`
- LongMemEval runner: `mem0/evaluation/benchmarks/longmemeval/run.py`
- Benchmark LLM client supports OpenAI-compatible `base_url`:
  - `mem0/evaluation/benchmarks/common/llm_client.py`
- Dataset download is built into the benchmark runners:
  - LoCoMo download in `benchmarks/locomo/run.py`
  - LongMemEval download in `benchmarks/longmemeval/run.py`

### Mem0 normal vs graph

- Current OSS Mem0 no longer exposes the old external graph toggle.
- Graph memory is now built-in entity linking.
- Relevant docs:
  - `mem0/docs/migration/oss-v2-to-v3.mdx`
  - `mem0/docs/platform/features/graph-memory.mdx`
  - `mem0/docs/core-concepts/memory-evaluation.mdx`

Operationally:

- With spaCy available, Mem0 uses entity extraction/linking.
- Without spaCy, Mem0 falls back to semantic-only retrieval.
- With Qdrant and `fastembed`, BM25 is also enabled.

So the closest reproducible split is:

- `Mem0-normal`: semantic-only setup
- `Mem0-graph`: built-in entity-linking setup

This is a setup distinction, not an official benchmark CLI flag.

### A-mem

- Core implementation is library-style and can be adapted:
  - `A-mem/agentic_memory/memory_system.py`
  - `A-mem/agentic_memory/llm_controller.py`
- No existing LoCoMo or LongMemEval benchmark runner found in repo.
- No existing Mem0 benchmark backend adapter for A-mem found in repo.

### MemoryBank

- Retrieval components exist, but the repo is not benchmark-shaped:
  - `MemoryBank-SiliconFriend/memory_bank/memory_retrieval/local_doc_qa.py`
  - `MemoryBank-SiliconFriend/memory_bank/memory_retrieval/forget_memory.py`
- Existing repo evaluation is its own probing-question setup, not LoCoMo or LongMemEval:
  - `MemoryBank-SiliconFriend/README.md`
  - `MemoryBank-SiliconFriend/eval_data/...`

## Patches added in this workspace

### Mem0 benchmark CLI

Added endpoint/key overrides to:

- `mem0/evaluation/benchmarks/locomo/run.py`
- `mem0/evaluation/benchmarks/longmemeval/run.py`

New flags:

- `--base-url`
- `--judge-base-url`
- `--api-key`
- `--judge-api-key`

### A-mem

Added OpenAI-compatible `base_url` support to:

- `A-mem/agentic_memory/llm_controller.py`
- `A-mem/agentic_memory/memory_system.py`

### MemoryBank

Added environment-driven endpoint/model support to:

- `MemoryBank-SiliconFriend/memory_bank/summarize_memory.py`
- `MemoryBank-SiliconFriend/memory_bank/build_memory_index.py`
- `MemoryBank-SiliconFriend/SiliconFriend-ChatGPT/cli_llamaindex.py`

Supported env vars:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `OPENAI_MODEL`

## What is runnable now

### Mem0 on LoCoMo / LongMemEval

Yes, with the current benchmark harness.

Example shape for NVIDIA NIM as answerer/judge:

```bash
conda run -n memory_eval312 python -m benchmarks.locomo.run \
  --project-name nim-mem0 \
  --provider openai \
  --base-url 'https://integrate.api.nvidia.com/v1' \
  --api-key "$NVIDIA_API_KEY"
```

```bash
conda run -n memory_eval312 python -m benchmarks.longmemeval.run \
  --project-name nim-mem0 \
  --provider openai \
  --base-url 'https://integrate.api.nvidia.com/v1' \
  --api-key "$NVIDIA_API_KEY" \
  --all-questions
```

These commands only cover the answerer/judge side. The Mem0 OSS server still needs its own LLM and embedder configuration.

## What is not yet runnable as a faithful benchmark

### A-mem on LoCoMo / LongMemEval

Not directly runnable yet.

Missing piece:

- A benchmark adapter that maps ingest/search calls from the benchmark suite onto A-mem.

### MemoryBank on LoCoMo / LongMemEval

Not directly runnable yet.

Missing pieces:

- A benchmark adapter
- A clear faithful mapping from benchmark ingest/search to MemoryBank's retrieval pipeline

## Recommendation

1. Run `Mem0-graph` first with NVIDIA NIM for answerer/judge.
2. Run `Mem0-normal` as a separate semantic-only setup.
3. Build an A-mem adapter next.
4. Treat MemoryBank as a higher-risk adapter effort.

