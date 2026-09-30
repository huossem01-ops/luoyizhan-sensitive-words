"""Retrieve romance-related real words/phrases from Tencent-style vectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml
from tqdm import tqdm

from candidate_filter import evaluate_candidate
from normalize import normalize_term


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "data" / "final"
REPORT_DIR = ROOT / "reports"
FIELDS = [
    "id", "term", "normalized_term", "length", "source", "score", "max_seed_similarity",
    "top3_seed_similarity", "best_seed", "best_category",
    "category_similarity", "semantic_level", "rank",
]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flatten_seeds(seed_config: dict) -> tuple[list[str], dict[str, list[str]]]:
    categories = seed_config["categories"]
    seeds = list(dict.fromkeys(seed for values in categories.values() for seed in values))
    return seeds, categories


class VectorSource:
    def __init__(self, path: Path, source_format: str):
        self.path = path
        self.source_format = source_format
        self.kv = None
        self.vocab_size = 0
        self.dimension = 0

    def open(self) -> "VectorSource":
        if self.source_format == "word2vec_binary":
            from gensim.models import KeyedVectors

            self.kv = KeyedVectors.load_word2vec_format(str(self.path), binary=True)
            self.vocab_size = len(self.kv)
            self.dimension = self.kv.vector_size
        elif self.source_format == "word2vec_text":
            with self.path.open(encoding="utf-8", errors="replace") as handle:
                header = handle.readline().split()
            self.vocab_size, self.dimension = map(int, header[:2])
        else:
            raise ValueError(f"unsupported source format: {self.source_format}")
        return self

    def seed_vectors(self, seeds: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        found = {}
        if self.kv is not None:
            for seed in seeds:
                if seed in self.kv:
                    found[seed] = self.kv.get_vector(seed, norm=True).astype(np.float32)
        else:
            wanted = set(seeds)
            for term, vector in self.iter_vectors(show_progress=False):
                if term in wanted:
                    found[term] = vector
                    if len(found) == len(wanted):
                        break
        return found, [seed for seed in seeds if seed not in found]

    def iter_vectors(self, show_progress: bool = True) -> Iterator[tuple[str, np.ndarray]]:
        if self.kv is not None:
            iterator = self.kv.index_to_key
            if show_progress:
                iterator = tqdm(iterator, total=self.vocab_size, desc="ranking vectors")
            for term in iterator:
                yield term, self.kv.get_vector(term, norm=True).astype(np.float32)
            return

        with self.path.open(encoding="utf-8", errors="replace") as handle:
            handle.readline()
            iterator = handle
            if show_progress:
                iterator = tqdm(handle, total=self.vocab_size, desc="ranking vectors")
            for line in iterator:
                fields = line.rstrip().split(" ")
                if len(fields) != self.dimension + 1:
                    continue
                try:
                    vector = np.asarray(fields[1:], dtype=np.float32)
                except ValueError:
                    continue
                norm = np.linalg.norm(vector)
                if norm:
                    yield fields[0], vector / norm


def build_seed_space(source: VectorSource, seed_config: dict) -> dict:
    unique_seeds, categories = flatten_seeds(seed_config)
    vectors, missing = source.seed_vectors(unique_seeds)
    if len(vectors) < 3:
        raise RuntimeError("可用 seed 少于 3 个，无法计算稳定的 top-3 similarity")
    present_seeds = [seed for seed in unique_seeds if seed in vectors]
    seed_matrix = np.stack([vectors[seed] for seed in present_seeds])

    category_names = []
    category_vectors = []
    category_seed_counts = {}
    for category, seeds in categories.items():
        available = [vectors[seed] for seed in dict.fromkeys(seeds) if seed in vectors]
        category_seed_counts[category] = len(available)
        if not available:
            continue
        centroid = np.mean(available, axis=0)
        centroid /= np.linalg.norm(centroid)
        category_names.append(category)
        category_vectors.append(centroid)
    return {
        "present_seeds": present_seeds,
        "missing_seeds": missing,
        "seed_matrix": seed_matrix,
        "category_names": category_names,
        "category_matrix": np.stack(category_vectors),
        "category_seed_counts": category_seed_counts,
    }


def provisional_level(rank: int, total: int, boundaries: dict) -> int:
    quantile = rank / total
    if quantile <= float(boundaries["level_5"]):
        return 5
    if quantile <= float(boundaries["level_4"]):
        return 4
    if quantile <= float(boundaries["level_3"]):
        return 3
    if quantile <= float(boundaries["level_2"]):
        return 2
    return 1


def retrieve(config: dict, seed_config: dict) -> tuple[list[dict], dict]:
    source_config = config["source"]
    source_path = ROOT / source_config["path"]
    if not source_path.exists():
        raise FileNotFoundError(f"向量文件不存在：{source_path}")
    source = VectorSource(source_path, source_config["format"]).open()
    seed_space = build_seed_space(source, seed_config)

    retrieval_config = config["retrieval"]
    top_k = int(retrieval_config["top_k"])
    pool_size = top_k * int(retrieval_config.get("pool_multiplier", 4))
    weights = retrieval_config["score_weights"]
    max_weight = float(weights["max_seed_similarity"])
    top3_weight = float(weights["top3_seed_similarity"])
    filter_counts = Counter()
    heap = []
    serial = 0

    for raw_term, vector in source.iter_vectors():
        decision = evaluate_candidate(raw_term, config["filters"])
        filter_counts[decision.reason] += 1
        if not decision.keep:
            continue
        similarities = seed_space["seed_matrix"] @ vector
        best_seed_index = int(np.argmax(similarities))
        max_similarity = float(similarities[best_seed_index])
        top3_mean = float(np.mean(np.partition(similarities, -3)[-3:]))
        category_similarities = seed_space["category_matrix"] @ vector
        best_category_index = int(np.argmax(category_similarities))
        score = max_weight * max_similarity + top3_weight * top3_mean
        record = {
            "term": decision.term,
            "normalized_term": normalize_term(decision.term),
            "length": decision.length,
            "source": source_config["name"],
            "score": score,
            "max_seed_similarity": max_similarity,
            "top3_seed_similarity": top3_mean,
            "best_seed": seed_space["present_seeds"][best_seed_index],
            "best_category": seed_space["category_names"][best_category_index],
            "category_similarity": float(category_similarities[best_category_index]),
        }
        item = (score, serial, record)
        serial += 1
        if len(heap) < pool_size:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    ranked_pool = [item[2] for item in sorted(heap, reverse=True)]
    unique = []
    seen = set()
    for record in ranked_pool:
        normalized = normalize_term(record["term"])
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(record)
        if len(unique) == top_k:
            break

    boundaries = config["levels"]["quantile_boundaries"]
    for index, record in enumerate(unique, start=1):
        record["id"] = index
        record["rank"] = index
        record["semantic_level"] = provisional_level(index, len(unique), boundaries)
        for key in ("score", "max_seed_similarity", "top3_seed_similarity", "category_similarity"):
            record[key] = round(record[key], 8)

    run = {
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_name": source_config["name"],
        "source_vocab_size": source.vocab_size,
        "vector_dimension": source.dimension,
        "requested_top_k": top_k,
        "returned": len(unique),
        "seed_count_configured": len(flatten_seeds(seed_config)[0]),
        "seed_count_in_vocab": len(seed_space["present_seeds"]),
        "seeds_in_vocab": seed_space["present_seeds"],
        "seeds_oov": seed_space["missing_seeds"],
        "category_seed_counts": seed_space["category_seed_counts"],
        "candidate_filter_counts": dict(filter_counts),
    }
    return unique, run


def write_outputs(records: list[dict], run: dict) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (FINAL_DIR / "sensitive_words.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (FINAL_DIR / "sensitive_words.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with (REPORT_DIR / "retrieval_run.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pipeline.yaml")
    parser.add_argument("--seeds", type=Path, default=ROOT / "config" / "seeds.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_config = load_yaml(args.seeds)
    records, run = retrieve(config, seed_config)
    write_outputs(records, run)
    print(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
