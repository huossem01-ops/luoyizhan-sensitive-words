"""Merge reviewed core terms with reviewed targeted Tencent expansions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "final" / "conservative_sensitive_words.jsonl"
EXPANSION = ROOT / "data" / "expansion" / "reranked_targeted_candidates.jsonl"
LABELS = ROOT / "reports" / "expansion_review_samples_labeled.csv"
OUTPUT_DIR = ROOT / "data" / "final"
FIELDS = [
    "id", "term", "normalized_term", "length", "source", "inclusion_route",
    "category", "best_seed", "tencent_similarity", "bge_similarity",
    "source_rank", "review_status",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    core = load_jsonl(CORE)
    expansion = load_jsonl(EXPANSION)
    with LABELS.open(encoding="utf-8-sig", newline="") as handle:
        keep_terms = {row["term"] for row in csv.DictReader(handle) if row["decision"] == "keep"}
    expansion_by_term = {row["term"]: row for row in expansion}

    output = []
    for row in core:
        output.append({
            "term": row["term"],
            "normalized_term": row["normalized_term"],
            "length": row["length"],
            "source": row["source"],
            "inclusion_route": "tencent_top5000_bge_fusion_reviewed",
            "category": row["best_category"],
            "best_seed": row["best_seed"],
            "tencent_similarity": row["tencent_score"],
            "bge_similarity": row["bge_score"],
            "source_rank": row["fusion_rank"],
            "review_status": "reviewed_keep",
        })
    for term in sorted(keep_terms, key=lambda value: expansion_by_term[value]["expansion_rank"]):
        row = expansion_by_term[term]
        output.append({
            "term": row["term"],
            "normalized_term": row["normalized_term"],
            "length": row["length"],
            "source": row["source"],
            "inclusion_route": "tencent_targeted_expansion_legacy_concept_audit",
            "category": row["target_category"],
            "best_seed": row["best_seed"],
            "tencent_similarity": row["target_score"],
            "bge_similarity": row["bge_target_score"],
            "source_rank": row["expansion_rank"],
            "review_status": "reviewed_keep",
        })
    if len({row["normalized_term"] for row in output}) != len(output):
        raise RuntimeError("normalized duplicate in expanded lexicon")
    for identifier, row in enumerate(output, start=1):
        row["id"] = identifier

    with (OUTPUT_DIR / "expanded_sensitive_words.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "expanded_sensitive_words.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)

    lengths = [row["length"] for row in output]
    report = {
        "total_records": len(output),
        "core_reviewed_records": len(core),
        "targeted_expansion_records": len(keep_terms),
        "unique_terms": len({row["term"] for row in output}),
        "unique_normalized_terms": len({row["normalized_term"] for row in output}),
        "average_length": round(float(np.mean(lengths)), 6),
        "median_length": float(np.median(lengths)),
        "p90_length": float(np.quantile(lengths, 0.9)),
        "max_length": max(lengths),
        "over_12": sum(length > 12 for length in lengths),
        "over_16": sum(length > 16 for length in lengths),
        "over_30": sum(length > 30 for length in lengths),
        "category_distribution": dict(Counter(row["category"] for row in output)),
        "route_distribution": dict(Counter(row["inclusion_route"] for row in output)),
    }
    with (ROOT / "reports" / "expanded_statistics.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
