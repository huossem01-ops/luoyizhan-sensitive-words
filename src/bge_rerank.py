"""Rerank the Tencent Top-5,000 candidate pool with BGE embeddings."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from tencent_retrieval import flatten_seeds, load_yaml


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "final" / "sensitive_words.jsonl"
OUTPUT_DIR = ROOT / "data" / "reranked"
REPORT_DIR = ROOT / "reports"


def load_records() -> list[dict]:
    with INPUT.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_or_load(texts: list[str], seeds: list[str], config: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    cache_path = ROOT / config["cache_path"]
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if cached["terms"].tolist() == texts and cached["seeds"].tolist() == seeds:
            return cached["term_embeddings"], cached["seed_embeddings"], True

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        config["model_name"],
        revision=config.get("revision"),
        device=config.get("device", "cpu"),
    )
    encode_options = {
        "batch_size": int(config.get("batch_size", 256)),
        "normalize_embeddings": True,
        "show_progress_bar": True,
        "convert_to_numpy": True,
    }
    # Terms and seeds are symmetric short expressions, so no query instruction
    # is used. This follows the BGE v1.5 model-card guidance for non-s2p tasks.
    term_embeddings = model.encode(texts, **encode_options).astype(np.float32)
    seed_embeddings = model.encode(seeds, **encode_options).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        terms=np.asarray(texts),
        seeds=np.asarray(seeds),
        term_embeddings=term_embeddings,
        seed_embeddings=seed_embeddings,
    )
    return term_embeddings, seed_embeddings, False


def rerank(records: list[dict], seed_config: dict, pipeline_config: dict) -> tuple[list[dict], dict]:
    config = pipeline_config["reranking"]
    seeds, categories = flatten_seeds(seed_config)
    terms = [record["term"] for record in records]
    term_embeddings, seed_embeddings, cache_hit = encode_or_load(terms, seeds, config)

    seed_similarities = term_embeddings @ seed_embeddings.T
    best_seed_indices = np.argmax(seed_similarities, axis=1)
    max_similarities = seed_similarities[np.arange(len(records)), best_seed_indices]
    top3_means = np.mean(np.partition(seed_similarities, -3, axis=1)[:, -3:], axis=1)

    seed_index = {seed: index for index, seed in enumerate(seeds)}
    category_names = list(categories)
    centroids = []
    for category in category_names:
        indices = [seed_index[seed] for seed in dict.fromkeys(categories[category])]
        centroid = seed_embeddings[indices].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        centroids.append(centroid)
    category_similarities = term_embeddings @ np.stack(centroids).T
    best_category_indices = np.argmax(category_similarities, axis=1)

    weights = pipeline_config["retrieval"]["score_weights"]
    scores = float(weights["max_seed_similarity"]) * max_similarities + float(weights["top3_seed_similarity"]) * top3_means
    order = np.argsort(-scores, kind="stable")[: int(config["output_top_k"])]
    output = []
    for new_rank, original_index in enumerate(order, start=1):
        original = records[int(original_index)]
        category_index = int(best_category_indices[original_index])
        output.append({
            "id": new_rank,
            "term": original["term"],
            "normalized_term": original["normalized_term"],
            "length": original["length"],
            "source": original["source"],
            "tencent_score": original["score"],
            "tencent_rank": original["rank"],
            "bge_score": round(float(scores[original_index]), 8),
            "bge_max_seed_similarity": round(float(max_similarities[original_index]), 8),
            "bge_top3_seed_similarity": round(float(top3_means[original_index]), 8),
            "bge_best_seed": seeds[int(best_seed_indices[original_index])],
            "bge_best_category": category_names[category_index],
            "bge_category_similarity": round(float(category_similarities[original_index, category_index]), 8),
            "bge_rank": new_rank,
        })
    run = {
        "model": config["model_name"],
        "revision": config.get("revision"),
        "device": config.get("device", "cpu"),
        "use_query_instruction": bool(config.get("use_query_instruction", False)),
        "candidate_count": len(records),
        "seed_count": len(seeds),
        "embedding_dimension": int(term_embeddings.shape[1]),
        "embedding_cache_hit": cache_hit,
        "bge_score_min": round(float(scores.min()), 8),
        "bge_score_max": round(float(scores.max()), 8),
    }
    return output, run


def write_outputs(records: list[dict], run: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / "bge_sensitive_words.jsonl"
    csv_path = OUTPUT_DIR / "bge_sensitive_words.csv"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (REPORT_DIR / "bge_rerank_run.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    sample_rows = []
    for center in (1, 1000, len(records)):
        if center == 1:
            subset = records[:100]
        elif center == len(records):
            subset = records[-100:]
        else:
            subset = records[center - 50:center + 50]
        for record in subset:
            sample_rows.append({
                "sample_center": center,
                "term": record["term"],
                "bge_score": record["bge_score"],
                "bge_rank": record["bge_rank"],
                "tencent_rank": record["tencent_rank"],
                "bge_best_seed": record["bge_best_seed"],
                "bge_best_category": record["bge_best_category"],
                "keep": "",
                "comment": "",
            })
    with (REPORT_DIR / "bge_manual_review_samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)


def main() -> None:
    records = load_records()
    seed_config = load_yaml(ROOT / "config" / "seeds.yaml")
    pipeline_config = load_yaml(ROOT / "config" / "pipeline.yaml")
    output, run = rerank(records, seed_config, pipeline_config)
    write_outputs(output, run)
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
