"""Hard validation gates for retrieval-based formal data."""

from __future__ import annotations

import json
from pathlib import Path

from candidate_filter import evaluate_candidate
from normalize import normalize_term


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "final" / "sensitive_words.jsonl"
REQUIRED = {
    "id", "term", "normalized_term", "length", "source", "score", "max_seed_similarity",
    "top3_seed_similarity", "best_seed", "best_category",
    "category_similarity", "semantic_level", "rank",
}


def main() -> None:
    with (ROOT / "config" / "pipeline.yaml").open(encoding="utf-8") as handle:
        import yaml
        config = yaml.safe_load(handle)
    with DATASET.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    errors = []
    terms = set()
    normalized = set()
    previous_score = float("inf")
    for index, record in enumerate(records, 1):
        if set(record) != REQUIRED:
            errors.append(f"record {index}: schema fields mismatch")
            continue
        if record["id"] != index or record["rank"] != index:
            errors.append(f"record {index}: id/rank must be consecutive")
        if record["term"] in terms:
            errors.append(f"record {index}: exact duplicate")
        terms.add(record["term"])
        normalized_term = normalize_term(record["term"])
        if record["normalized_term"] != normalized_term:
            errors.append(f"record {index}: normalized_term mismatch")
        if normalized_term in normalized:
            errors.append(f"record {index}: normalized duplicate")
        normalized.add(normalized_term)
        if record["length"] != len(record["term"]):
            errors.append(f"record {index}: length mismatch")
        if record["length"] > 30:
            errors.append(f"record {index}: term over 30 characters")
        decision = evaluate_candidate(record["term"], config["filters"])
        if not decision.keep:
            errors.append(f"record {index}: candidate filter failed: {decision.reason}")
        if not 1 <= record["semantic_level"] <= 5:
            errors.append(f"record {index}: invalid semantic_level")
        if record["score"] > previous_score + 1e-8:
            errors.append(f"record {index}: scores not descending")
        previous_score = record["score"]
        if "legacy" in record["source"].lower():
            errors.append(f"record {index}: legacy source in formal data")
    if errors:
        print(f"validation=failed records={len(records)} errors={len(errors)}")
        for error in errors[:100]:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"validation=passed records={len(records)} max_length={max(r['length'] for r in records)}")


if __name__ == "__main__":
    main()
