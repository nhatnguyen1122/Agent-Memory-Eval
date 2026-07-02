#!/usr/bin/env python3
"""Smoke-test a single LongMemEval ingestion pair against a memory backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from benchmarks.common.memory_backends import create_memory_backend
from benchmarks.longmemeval.run import parse_longmemeval_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="mem0", choices=["mem0", "mem0_graph", "a_mem", "memorybank"])
    parser.add_argument("--question-id", default="311778f1")
    parser.add_argument("--session-idx", type=int, default=0)
    parser.add_argument("--pair-idx", type=int, default=0)
    parser.add_argument("--api-key", default=os.getenv("NVIDIA_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    parser.add_argument("--memory-model", default=os.getenv("MEMORY_MODEL", "meta/llama-3.1-8b-instruct"))
    parser.add_argument("--memory-embedder-model", default=os.getenv("MEMORY_EMBEDDER_MODEL", "nvidia/llama-nemotron-embed-1b-v2"))
    parser.add_argument("--storage-dir", default="mem0/evaluation/.benchmark_state_pair_smoke")
    parser.add_argument("--dataset", default="mem0/evaluation/datasets/longmemeval/longmemeval_s_cleaned.json")
    return parser.parse_args()


def pair_turns(session: list[dict]) -> list[list[dict]]:
    cleaned = [{"role": t["role"], "content": t["content"]} for t in session]
    return [cleaned[i : i + 2] for i in range(0, len(cleaned), 2)]


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

    questions = json.loads(Path(args.dataset).read_text())
    question = next(q for q in questions if q["question_id"] == args.question_id)
    session = question["haystack_sessions"][args.session_idx]
    session_id = question["haystack_session_ids"][args.session_idx]
    date_str = question["haystack_dates"][args.session_idx]
    messages = pair_turns(session)[args.pair_idx]
    metadata = {
        "benchmark": "longmemeval",
        "question_id": question["question_id"],
        "question_type": question.get("question_type", ""),
        "source_session_id": session_id,
        "source_session_idx": args.session_idx,
        "source_pair_idx": args.pair_idx,
        "source_date": date_str,
        "answer_session_ids": question.get("answer_session_ids", []),
    }

    class _Args:
        project_name = f"pair-smoke-{args.question_id}-{args.backend}"
        memory_backend = args.backend
        memory_api_key = args.api_key
        memory_base_url = args.base_url
        memory_model = args.memory_model
        memory_embedder_model = args.memory_embedder_model
        memory_storage_dir = args.storage_dir
        backend = "oss"
        mem0_host = None
        mem0_api_key = None

    print("backend", args.backend)
    print("question", question["question_id"], question["question_type"])
    print("session", session_id, date_str)
    print("roles", [m["role"] for m in messages])
    print("chars", [len(m["content"]) for m in messages])
    print("embedder", args.memory_embedder_model)

    backend = create_memory_backend(_Args(), benchmark_name="pair_smoke", output_dir="")
    async with backend:
        result = await backend.add(
            messages,
            user_id=f"pair_smoke_{args.question_id}",
            timestamp=parse_longmemeval_date(date_str),
            metadata=metadata,
        )
        print("ADD_RESULT", json.dumps(result, ensure_ascii=False)[:1000])
        search = await backend.search(question["question"], user_id=f"pair_smoke_{args.question_id}", top_k=5)
        print("SEARCH_HITS", len(search))
        if search:
            print("TOP", json.dumps(search[0], ensure_ascii=False)[:1000])


if __name__ == "__main__":
    asyncio.run(main())
