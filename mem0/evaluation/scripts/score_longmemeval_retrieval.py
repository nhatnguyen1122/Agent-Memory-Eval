#!/usr/bin/env python3
"""Compute exact LongMemEval source-session retrieval metrics.

This scorer expects result JSONs whose retrieval search results carry source
session provenance, added by the multi-backend harness. Older result files that
do not include source ids are reported with missing_source_ids and cannot
produce reliable exact recall.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_cutoffs(value: str) -> list[int]:
    cutoffs = []
    for item in value.split(","):
        item = item.strip()
        if item:
            cutoffs.append(int(item))
    return sorted(set(cutoffs))


def iter_json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
    return files


def infer_backend(path: Path, metadata: dict[str, Any] | None = None) -> str:
    if metadata and metadata.get("memory_backend"):
        return str(metadata["memory_backend"])
    if metadata and metadata.get("project_name"):
        project_name = str(metadata["project_name"])
        for backend in ("mem0_graph", "memorybank", "a_mem", "mem0", "mem0_server"):
            if project_name.endswith(f"-{backend}") or f"-{backend}-" in project_name:
                return backend
    for part in reversed(path.parts):
        if part.startswith("nim-longmemeval-small-"):
            return part.removeprefix("nim-longmemeval-small-")
        if part in {"mem0", "mem0_graph", "a_mem", "memorybank", "mem0_server"}:
            return part
    return "unknown"


def load_evaluations(files: list[Path]) -> list[tuple[Path, str, dict[str, Any]]]:
    parsed: list[tuple[Path, dict[str, Any]]] = []

    for path in files:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            parsed.append((path, data))

    loaded: list[tuple[Path, str, dict[str, Any]]] = []
    unified_files = [
        (path, data) for path, data in parsed if isinstance(data.get("evaluations"), list)
    ]
    source_files = unified_files or parsed

    for path, data in source_files:
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(data, dict) and isinstance(data.get("evaluations"), list):
            backend = infer_backend(path, metadata)
            for evaluation in data["evaluations"]:
                loaded.append((path, backend, evaluation))
        elif isinstance(data, dict) and data.get("question_id"):
            loaded.append((path, infer_backend(path), data))

    return loaded


def add_source_id(ids: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, list):
        for item in value:
            add_source_id(ids, item)
        return
    text = str(value)
    if text:
        ids.append(text)


def source_session_ids(result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    add_source_id(ids, result.get("source_session_id"))
    add_source_id(ids, result.get("source_session_ids"))

    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        add_source_id(ids, metadata.get("source_session_id"))
        add_source_id(ids, metadata.get("source_session_ids"))
        for entry in metadata.get("source_entries") or []:
            if isinstance(entry, dict):
                add_source_id(ids, entry.get("source_session_id"))
                add_source_id(ids, entry.get("source_session_ids"))

    seen = set()
    deduped = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def result_rows(
    evaluations: list[tuple[Path, str, dict[str, Any]]],
    cutoffs: list[int],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "hit_sum": 0.0,
            "recall_sum": 0.0,
            "precision_sum": 0.0,
            "missing_source_ids": 0,
            "no_gold": 0,
        }
    )

    for _path, backend, evaluation in evaluations:
        qtype = str(evaluation.get("question_type", "unknown"))
        gold = {str(x) for x in evaluation.get("answer_session_ids", []) if str(x)}
        search_results = (evaluation.get("retrieval") or {}).get("search_results", [])

        for cutoff in cutoffs:
            bucket = buckets[(backend, qtype, cutoff)]
            bucket["n"] += 1
            if not gold:
                bucket["no_gold"] += 1
                continue

            retrieved: list[str] = []
            missing_for_question = False
            for result in search_results[:cutoff]:
                ids = source_session_ids(result)
                if not ids:
                    missing_for_question = True
                retrieved.extend(ids)

            retrieved_set = set(retrieved)
            overlap = retrieved_set & gold
            bucket["hit_sum"] += 1.0 if overlap else 0.0
            bucket["recall_sum"] += len(overlap) / len(gold)
            bucket["precision_sum"] += len(overlap) / max(cutoff, 1)
            if missing_for_question:
                bucket["missing_source_ids"] += 1

    rows = []
    for (backend, qtype, cutoff), data in sorted(buckets.items()):
        n = data["n"]
        rows.append(
            {
                "backend": backend,
                "question_type": qtype,
                "k": cutoff,
                "n": n,
                "hit@k": data["hit_sum"] / n if n else 0.0,
                "recall@k": data["recall_sum"] / n if n else 0.0,
                "precision@k": data["precision_sum"] / n if n else 0.0,
                "missing_source_ids": data["missing_source_ids"],
                "no_gold": data["no_gold"],
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], output: str) -> None:
    if not rows:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No LongMemEval evaluations found.")
        return

    print("backend, question_type, k, n, hit@k, recall@k, precision@k, missing_source_ids")
    for row in rows:
        print(
            f"{row['backend']}, {row['question_type']}, {row['k']}, {row['n']}, "
            f"{row['hit@k']:.3f}, {row['recall@k']:.3f}, {row['precision@k']:.3f}, "
            f"{row['missing_source_ids']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="LongMemEval unified result JSON(s) or directories containing result JSONs.",
    )
    parser.add_argument("--cutoffs", default="1,5,10,20,50")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    files = iter_json_files(args.paths)
    evaluations = load_evaluations(files)
    rows = result_rows(evaluations, parse_cutoffs(args.cutoffs))
    print_rows(rows)
    if args.output:
        write_csv(rows, args.output)
        print(f"\nSaved CSV to: {args.output}")


if __name__ == "__main__":
    main()
