"""Build the conservative, explicitly reviewed subset from fusion results."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FUSION = ROOT / "data" / "reranked" / "fused_sensitive_words.jsonl"
LABEL_FILES = (
    ROOT / "reports" / "manual_review_samples_labeled.csv",
    ROOT / "reports" / "bge_manual_review_samples_labeled.csv",
)
HOLDOUT = ROOT / "reports" / "fusion_holdout_review.csv"
OUTPUT_DIR = ROOT / "data" / "final"


def main() -> None:
    decisions: dict[str, str] = {}
    for path in LABEL_FILES:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                previous = decisions.get(row["term"])
                if previous is not None and previous != row["keep"]:
                    raise RuntimeError(f"conflicting label for {row['term']}")
                decisions[row["term"]] = row["keep"]
    with HOLDOUT.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            decisions[row["term"]] = row["decision"]

    with FUSION.open(encoding="utf-8") as handle:
        fusion = [json.loads(line) for line in handle if line.strip()]
    selected = [row for row in fusion if decisions.get(row["term"]) == "keep"]
    for rank, row in enumerate(selected, start=1):
        row["id"] = rank
        row["rank"] = rank
        row["review_status"] = "reviewed_keep"

    fields = list(selected[0])
    with (OUTPUT_DIR / "conservative_sensitive_words.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "conservative_sensitive_words.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    lengths = [row["length"] for row in selected]
    report = {
        "reviewed_keep_count": len(selected),
        "review_decision_distribution": dict(Counter(decisions.values())),
        "unique_terms": len({row["term"] for row in selected}),
        "average_length": round(float(np.mean(lengths)), 6),
        "median_length": float(np.median(lengths)),
        "p90_length": float(np.quantile(lengths, 0.9)),
        "max_length": max(lengths),
        "over_12": sum(length > 12 for length in lengths),
        "over_16": sum(length > 16 for length in lengths),
        "over_30": sum(length > 30 for length in lengths),
        "note": "Only first-pass reviewed keep labels are included; borderline and unreviewed candidates are excluded.",
    }
    with (ROOT / "reports" / "conservative_statistics.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
