"""BGE reranking for targeted Tencent expansion candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tencent_retrieval import load_yaml


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "expansion" / "tencent_targeted_candidates.jsonl"
OUTPUT_DIR = ROOT / "data" / "expansion"
REPORTS = ROOT / "reports"
CACHE = ROOT / "cache" / "bge" / "targeted_expansion.npz"


def main() -> None:
    pipeline = load_yaml(ROOT / "config" / "pipeline.yaml")
    expansion = load_yaml(ROOT / "config" / "expansion_seeds.yaml")
    with INPUT.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    terms = [record["term"] for record in records]
    seeds = list(dict.fromkeys(seed for values in expansion["categories"].values() for seed in values))

    cache_hit = False
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=False)
        if cached["terms"].tolist() == terms and cached["seeds"].tolist() == seeds:
            term_embeddings = cached["term_embeddings"]
            seed_embeddings = cached["seed_embeddings"]
            cache_hit = True
    if not cache_hit:
        from sentence_transformers import SentenceTransformer

        config = pipeline["reranking"]
        model = SentenceTransformer(config["model_name"], revision=config["revision"], device=config.get("device", "cpu"))
        options = {
            "batch_size": int(config.get("batch_size", 256)),
            "normalize_embeddings": True,
            "show_progress_bar": True,
            "convert_to_numpy": True,
        }
        term_embeddings = model.encode(terms, **options).astype(np.float32)
        seed_embeddings = model.encode(seeds, **options).astype(np.float32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE,
            terms=np.asarray(terms), seeds=np.asarray(seeds),
            term_embeddings=term_embeddings, seed_embeddings=seed_embeddings,
        )

    seed_index = {seed: index for index, seed in enumerate(seeds)}
    weights = expansion["retrieval"]["score_weights"]
    for index, record in enumerate(records):
        category_seeds = expansion["categories"][record["target_category"]]
        indices = [seed_index[seed] for seed in category_seeds]
        similarities = term_embeddings[index] @ seed_embeddings[indices].T
        best = int(np.argmax(similarities))
        take = min(3, len(similarities))
        max_similarity = float(similarities[best])
        top_mean = float(np.mean(np.partition(similarities, -take)[-take:]))
        record["bge_target_score"] = round(
            float(weights["max_seed_similarity"]) * max_similarity
            + float(weights["top3_seed_similarity"]) * top_mean,
            8,
        )
        record["bge_best_seed"] = category_seeds[best]

    by_category: dict[str, list[dict]] = {category: [] for category in expansion["categories"]}
    for record in records:
        by_category[record["target_category"]].append(record)
    for category, values in by_category.items():
        for rank, record in enumerate(sorted(values, key=lambda row: -row["target_score"]), start=1):
            record["tencent_category_rank"] = rank
        for rank, record in enumerate(sorted(values, key=lambda row: -row["bge_target_score"]), start=1):
            record["bge_category_rank"] = rank
        for record in values:
            record["rank_fusion_score"] = round(
                0.5 / (60 + record["tencent_category_rank"])
                + 0.5 / (60 + record["bge_category_rank"]),
                10,
            )

    reranked = sorted(records, key=lambda row: (-row["rank_fusion_score"], row["term"]))
    for rank, record in enumerate(reranked, start=1):
        record["expansion_rank"] = rank
        record["id"] = rank
        record["rank"] = rank
    fields = list(reranked[0])
    with (OUTPUT_DIR / "reranked_targeted_candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in reranked:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "reranked_targeted_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reranked)

    review = []
    for category, values in by_category.items():
        ranked = sorted(values, key=lambda row: -row["rank_fusion_score"])
        category_seed_terms = set(expansion["categories"][category])
        selected = ranked[:40]
        selected_terms = {row["term"] for row in selected}
        selected.extend(
            row for row in ranked
            if row["term"] in category_seed_terms and row["term"] not in selected_terms
        )
        for record in selected:
            review.append({
                "target_category": category,
                "term": record["term"],
                "target_score": record["target_score"],
                "bge_target_score": record["bge_target_score"],
                "rank_fusion_score": record["rank_fusion_score"],
                "best_seed": record["best_seed"],
                "bge_best_seed": record["bge_best_seed"],
                "legacy_exact_reference": record["legacy_exact_reference"],
                "decision": "",
                "comment": "",
            })
    with (REPORTS / "expansion_review_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(review)
    run = {
        "candidate_count": len(records),
        "review_sample_count": len(review),
        "review_per_category": "top40_plus_novel_in_vocab_seed_terms",
        "embedding_cache_hit": cache_hit,
        "model": pipeline["reranking"]["model_name"],
        "revision": pipeline["reranking"]["revision"],
    }
    with (REPORTS / "expansion_rerank_run.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
