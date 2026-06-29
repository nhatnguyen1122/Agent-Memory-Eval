import argparse
import asyncio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from benchmarks.common.memory_backends import create_memory_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a benchmark memory backend.")
    parser.add_argument("--backend", required=True, choices=["mem0", "mem0_graph", "a_mem", "memorybank"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--memory-model", default=None)
    parser.add_argument("--memory-embedder-model", default=None)
    parser.add_argument("--storage-dir", default=".benchmark_state_smoke")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    class _Args:
        project_name = "backend-smoke"
        memory_backend = args.backend
        memory_api_key = args.api_key
        memory_base_url = args.base_url
        memory_model = args.memory_model
        memory_embedder_model = args.memory_embedder_model
        memory_storage_dir = str(storage_dir)
        backend = "oss"
        mem0_host = None
        mem0_api_key = None

    backend = create_memory_backend(
        _Args(),
        benchmark_name="smoke",
        output_dir=storage_dir / "output",
    )

    user_id = f"smoke_{args.backend}"
    memories = [
        [{"role": "user", "content": "On 2026-01-10 my favorite dessert is tiramisu."}],
        [{"role": "user", "content": "On 2026-02-20 I said I am planning a trip to Kyoto in October."}],
    ]
    for memory in memories:
        await backend.add(memory, user_id=user_id)

    results = await backend.search("What dessert does the user like?", user_id=user_id, top_k=5)
    print(f"backend={args.backend}")
    print(f"hits={len(results)}")
    if results:
        first = results[0]
        if isinstance(first, dict):
            text = first.get("memory") or first.get("text") or str(first)
        else:
            text = str(first)
        print(f"top_result={text[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
