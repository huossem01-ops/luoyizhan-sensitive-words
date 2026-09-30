"""Generate deterministic raw JSONL batches from the semantic catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog import iter_candidates
from expansion_catalog import (
    ALL_SPECS,
    TARGET_COUNTS,
    iter_expansion_candidates,
    iter_new_seeds,
)
from normalize import normalize_term


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def build_candidate_pool() -> list[dict]:
    """Build exactly 10k candidates while preserving high-precision seeds."""
    pools: dict[str, list[dict]] = {category: [] for category in ALL_SPECS}
    seen = set()

    for candidate in iter_candidates():
        normalized = normalize_term(candidate["term"])
        if candidate["category"] in pools and normalized not in seen:
            pools[candidate["category"]].append(candidate)
            seen.add(normalized)

    for candidate in iter_new_seeds():
        normalized = normalize_term(candidate["term"])
        if normalized not in seen:
            pools[candidate["category"]].append(candidate)
            seen.add(normalized)

    for category, target in TARGET_COUNTS.items():
        for candidate in iter_expansion_candidates(category):
            if len(pools[category]) >= target:
                break
            normalized = normalize_term(candidate["term"])
            if normalized in seen:
                continue
            pools[category].append(candidate)
            seen.add(normalized)
        if len(pools[category]) < target:
            raise RuntimeError(
                f"类别 {category} 只有 {len(pools[category])} 条，低于目标 {target}"
            )

    # Round-robin output keeps prefixes useful for small development samples.
    ordered = []
    width = max(map(len, pools.values()))
    for index in range(width):
        for category in ALL_SPECS:
            if index < len(pools[category]):
                ordered.append(pools[category][index])
    return ordered


def build_records(count: int) -> list[dict]:
    candidates = build_candidate_pool()
    if count < 1 or count > len(candidates):
        raise ValueError(f"count 必须在 1 到 {len(candidates)} 之间")

    records = []
    for identifier, candidate in enumerate(candidates[:count], start=1):
        record = {"id": identifier, **candidate}
        record["normalized_term"] = normalize_term(record["term"])
        # Keep schema field order stable for readable diffs.
        records.append({
            "id": record["id"],
            "term": record["term"],
            "normalized_term": record["normalized_term"],
            "category": record["category"],
            "subcategory": record["subcategory"],
            "trigger_level": record["trigger_level"],
            "semantic_distance": record["semantic_distance"],
            "humor": record["humor"],
            "reason": record["reason"],
            "tags": record["tags"],
        })
    return records


def write_batches(records: list[dict], batch_size: int) -> list[Path]:
    if batch_size < 1:
        raise ValueError("batch-size 必须大于 0")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for old_batch in RAW_DIR.glob("batch_*.jsonl"):
        old_batch.unlink()

    paths = []
    for start in range(0, len(records), batch_size):
        path = RAW_DIR / f"batch_{start // batch_size + 1:03d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records[start:start + batch_size]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    records = build_records(args.count)
    paths = write_batches(records, args.batch_size)
    print(f"generated={len(records)} batches={len(paths)} raw_dir={RAW_DIR}")


if __name__ == "__main__":
    main()
