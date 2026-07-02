#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, reference: str) -> float:
    pred_tokens = tokens(prediction)
    ref_tokens = tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens)
    if precision == 0:
        return 0.0
    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return brevity_penalty * precision


def backend_from_project(project_name: str) -> str:
    for backend in ("mem0_graph", "memorybank", "a_mem", "mem0"):
        if project_name.endswith(f"-{backend}") or project_name.endswith(backend):
            return backend
    return project_name


def summarize_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    metadata = data.get("metadata", {})
    project_name = metadata.get("project_name", path.stem)
    backend = backend_from_project(project_name)
    evaluations = data.get("evaluations", [])

    by_cutoff: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "f1": [],
            "bleu1": [],
            "judge_score": [],
            "categories": set(),
        }
    )

    for item in evaluations:
        reference = str(item.get("ground_truth_answer", ""))
        category = item.get("category_name", "unknown")
        for cutoff, cutoff_result in item.get("cutoff_results", {}).items():
            prediction = str(cutoff_result.get("generated_answer", ""))
            by_cutoff[cutoff]["f1"].append(token_f1(prediction, reference))
            by_cutoff[cutoff]["bleu1"].append(bleu1(prediction, reference))
            by_cutoff[cutoff]["judge_score"].append(float(cutoff_result.get("score", 0.0)))
            by_cutoff[cutoff]["categories"].add(category)

    rows = []
    for cutoff in sorted(by_cutoff, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0):
        values = by_cutoff[cutoff]
        n = len(values["f1"])
        rows.append(
            {
                "file": str(path),
                "project_name": project_name,
                "backend": backend,
                "category": "+".join(sorted(values["categories"])),
                "cutoff": cutoff,
                "n": n,
                "f1": 100 * sum(values["f1"]) / n if n else 0.0,
                "bleu1": 100 * sum(values["bleu1"]) / n if n else 0.0,
                "judge": 100 * sum(values["judge_score"]) / n if n else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LoCoMo token F1 and BLEU-1 from saved result JSONs.")
    parser.add_argument("paths", nargs="+", help="Result JSON files or directories to scan.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    result_files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            result_files.extend(sorted(path.glob("**/locomo_results_*.json")))
        else:
            result_files.append(path)

    rows: list[dict[str, Any]] = []
    for path in result_files:
        rows.extend(summarize_file(path))

    rows.sort(key=lambda r: (r["category"], r["backend"], r["cutoff"], r["project_name"]))

    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["project_name", "backend", "category", "cutoff", "n", "f1", "bleu1", "judge", "file"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print("project_name,backend,category,cutoff,n,F1,BLEU1,J")
    for row in rows:
        print(
            f"{row['project_name']},{row['backend']},{row['category']},{row['cutoff']},"
            f"{row['n']},{row['f1']:.2f},{row['bleu1']:.2f},{row['judge']:.2f}"
        )


if __name__ == "__main__":
    main()
