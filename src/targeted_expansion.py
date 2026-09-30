"""Targeted Tencent-vector expansion from legacy-discovered concept gaps.

Legacy generated terms are read only to record overlap/provenance. Every output
candidate must exist in the configured Tencent vector vocabulary.
"""

from __future__ import annotations

import csv
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from candidate_filter import evaluate_candidate
from normalize import normalize_term
from tencent_retrieval import VectorSource, load_yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "expansion"
REPORT_DIR = ROOT / "reports"


def load_terms(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["term"] for line in handle if line.strip()}


def main() -> None:
    pipeline = load_yaml(ROOT / "config" / "pipeline.yaml")
    expansion = load_yaml(ROOT / "config" / "expansion_seeds.yaml")
    source_config = pipeline["source"]
    source = VectorSource(ROOT / source_config["path"], source_config["format"]).open()
    existing = load_terms(ROOT / "data" / "final" / "sensitive_words.jsonl")
    legacy = load_terms(ROOT / "data" / "legacy_generated" / "sensitive_words.jsonl")

    categories = expansion["categories"]
    seed_status = {}
    category_spaces = {}
    for category, seeds in categories.items():
        vectors, missing = source.seed_vectors(seeds)
        present = [seed for seed in seeds if seed in vectors]
        if not present:
            raise RuntimeError(f"category has no in-vocabulary seeds: {category}")
        category_spaces[category] = (present, np.stack([vectors[seed] for seed in present]))
        seed_status[category] = {"in_vocab": present, "oov": missing}

    retrieval = expansion["retrieval"]
    per_category_k = int(retrieval["per_category_k"])
    pool_size = per_category_k * int(retrieval.get("pool_multiplier", 2))
    min_score = float(retrieval["min_score"])
    weights = retrieval["score_weights"]
    max_weight = float(weights["max_seed_similarity"])
    top_weight = float(weights["top3_seed_similarity"])
    heaps: dict[str, list[tuple[float, int, dict]]] = {category: [] for category in categories}
    filter_counts = Counter()
    serial = 0

    for raw_term, vector in tqdm(source.iter_vectors(show_progress=False), total=source.vocab_size, desc="targeted expansion"):
        decision = evaluate_candidate(raw_term, pipeline["filters"])
        filter_counts[decision.reason] += 1
        if not decision.keep or decision.term in existing:
            continue
        for category, (seeds, matrix) in category_spaces.items():
            similarities = matrix @ vector
            best_index = int(np.argmax(similarities))
            max_similarity = float(similarities[best_index])
            take = min(3, len(similarities))
            top_mean = float(np.mean(np.partition(similarities, -take)[-take:]))
            score = max_weight * max_similarity + top_weight * top_mean
            if score < min_score:
                continue
            record = {
                "term": decision.term,
                "normalized_term": normalize_term(decision.term),
                "length": decision.length,
                "source": source_config["name"],
                "target_category": category,
                "target_score": score,
                "max_seed_similarity": max_similarity,
                "top3_seed_similarity": top_mean,
                "best_seed": seeds[best_index],
                "legacy_exact_reference": decision.term in legacy,
            }
            item = (score, serial, record)
            serial += 1
            heap = heaps[category]
            if len(heap) < pool_size:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    combined: dict[str, dict] = {}
    per_category_counts = {}
    for category, heap in heaps.items():
        selected, seen = [], set()
        for _, _, record in sorted(heap, reverse=True):
            normalized = record["normalized_term"]
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(record)
            if len(selected) == per_category_k:
                break
        per_category_counts[category] = len(selected)
        for record in selected:
            current = combined.get(record["normalized_term"])
            if current is None or record["target_score"] > current["target_score"]:
                combined[record["normalized_term"]] = record

    # A concept-discovery seed can be penalized by the top-3 stability term and
    # fall outside its category quota. Preserve every real in-vocabulary novel
    # seed in the audit pool so important gaps such as “刘海” are not lost again.
    for category, (seeds, matrix) in category_spaces.items():
        for seed in seeds:
            if seed in existing:
                continue
            normalized = normalize_term(seed)
            if normalized in combined:
                continue
            vector = source.kv.get_vector(seed, norm=True).astype(np.float32)
            similarities = matrix @ vector
            best_index = int(np.argmax(similarities))
            take = min(3, len(similarities))
            max_similarity = float(similarities[best_index])
            top_mean = float(np.mean(np.partition(similarities, -take)[-take:]))
            score = max_weight * max_similarity + top_weight * top_mean
            combined[normalized] = {
                "term": seed,
                "normalized_term": normalized,
                "length": len(seed),
                "source": source_config["name"],
                "target_category": category,
                "target_score": score,
                "max_seed_similarity": max_similarity,
                "top3_seed_similarity": top_mean,
                "best_seed": seeds[best_index],
                "legacy_exact_reference": seed in legacy,
            }

    records = sorted(combined.values(), key=lambda row: (-row["target_score"], row["term"]))
    for rank, record in enumerate(records, start=1):
        record["id"] = rank
        record["rank"] = rank
        for key in ("target_score", "max_seed_similarity", "top3_seed_similarity"):
            record[key] = round(float(record[key]), 8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "tencent_targeted_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "tencent_targeted_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "policy": expansion["policy"],
        "source_vocab_size": source.vocab_size,
        "excluded_existing_top5000": len(existing),
        "seed_status": seed_status,
        "per_category_requested": per_category_k,
        "per_category_retrieved_before_cross_category_dedup": per_category_counts,
        "unique_incremental_candidates": len(records),
        "legacy_exact_reference_count": sum(row["legacy_exact_reference"] for row in records),
        "filter_counts": dict(filter_counts),
    }
    with (REPORT_DIR / "targeted_expansion_run.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
