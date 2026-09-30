"""Generate QA statistics, distributions, and manual-review samples."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from normalize import normalize_term


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "final" / "sensitive_words.jsonl"
REPORT_DIR = ROOT / "reports"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with DATASET.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    with (ROOT / "config" / "pipeline.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(records)
    lengths = [record["length"] for record in records]
    scores = np.asarray([record["score"] for record in records], dtype=float)

    length_counts = Counter(lengths)
    write_rows(
        REPORT_DIR / "length_distribution.csv",
        ["length", "count", "share"],
        [{"length": length, "count": count, "share": round(count / total, 8)} for length, count in sorted(length_counts.items())],
    )
    category_counts = Counter(record["best_category"] for record in records)
    write_rows(
        REPORT_DIR / "category_distribution.csv",
        ["category", "count", "share"],
        [{"category": category, "count": count, "share": round(count / total, 8)} for category, count in category_counts.most_common()],
    )

    duplicate_rows = []
    duplicate_path = REPORT_DIR / "duplicates_report.csv"
    if duplicate_path.exists():
        with duplicate_path.open(encoding="utf-8-sig", newline="") as handle:
            duplicate_rows = list(csv.DictReader(handle))
    exact_or_normalized = sum(row["match_type"] in {"exact", "normalized"} for row in duplicate_rows)

    qa_config = config["qa"]
    rng = random.Random(int(qa_config["random_seed"]))
    sample_size = min(int(qa_config["random_sample_size"]), total)
    random_samples = rng.sample(records, sample_size)
    sample_fields = ["term", "length", "score", "rank", "best_seed", "best_category", "semantic_level"]
    write_rows(REPORT_DIR / "random_samples.csv", sample_fields, [{key: row[key] for key in sample_fields} for row in random_samples])

    manual_rows = []
    unavailable_centers = []
    window_size = int(qa_config["manual_sample_size"])
    for center in qa_config["manual_review_centers"]:
        center = int(center)
        if center > total:
            unavailable_centers.append(center)
            continue
        if center == 1:
            start, end = 1, min(total, window_size)
        elif center == total:
            start, end = max(1, total - window_size + 1), total
        else:
            start = max(1, center - window_size // 2 + 1)
            end = min(total, start + window_size - 1)
            start = max(1, end - window_size + 1)
        for record in records[start - 1:end]:
            manual_rows.append({
                "sample_center": center,
                "term": record["term"],
                "score": record["score"],
                "rank": record["rank"],
                "best_seed": record["best_seed"],
                "best_category": record["best_category"],
                "keep": "",
                "comment": "",
            })
    write_rows(
        REPORT_DIR / "manual_review_samples.csv",
        ["sample_center", "term", "score", "rank", "best_seed", "best_category", "keep", "comment"],
        manual_rows,
    )

    level_counts = Counter(record["semantic_level"] for record in records)
    statistics_report = {
        "total_records": total,
        "unique_terms": len({record["term"] for record in records}),
        "unique_normalized_terms": len({normalize_term(record["term"]) for record in records}),
        "exact_or_normalized_duplicate_pairs": exact_or_normalized,
        "fuzzy_duplicate_candidates": sum(row["match_type"] == "fuzzy" for row in duplicate_rows),
        "average_length": round(float(np.mean(lengths)), 6),
        "median_length": float(np.median(lengths)),
        "p90_length": float(np.quantile(lengths, 0.90)),
        "max_length": max(lengths),
        "ratio_over_12": round(sum(length > 12 for length in lengths) / total, 8),
        "ratio_over_16": round(sum(length > 16 for length in lengths) / total, 8),
        "ratio_over_30": round(sum(length > 30 for length in lengths) / total, 8),
        "category_distribution": dict(category_counts),
        "similarity_distribution": {
            "min": round(float(scores.min()), 8),
            "max": round(float(scores.max()), 8),
            "mean": round(float(scores.mean()), 8),
            "p10": round(float(np.quantile(scores, 0.10)), 8),
            "p25": round(float(np.quantile(scores, 0.25)), 8),
            "median": round(float(np.quantile(scores, 0.50)), 8),
            "p75": round(float(np.quantile(scores, 0.75)), 8),
            "p90": round(float(np.quantile(scores, 0.90)), 8),
        },
        "level_distribution": {str(level): level_counts[level] for level in sorted(level_counts)},
        "manual_review_centers_unavailable_in_current_top_k": unavailable_centers,
    }
    with (REPORT_DIR / "statistics.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(statistics_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(statistics_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
