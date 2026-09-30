"""Validate the reviewed expanded lexicon and its Tencent provenance."""

from __future__ import annotations

import json
from pathlib import Path

from gensim.models import KeyedVectors

from normalize import normalize_term


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "final" / "expanded_sensitive_words.jsonl"
SCHEMA = ROOT / "schema" / "expanded_data.schema.json"
VECTOR_PATH = ROOT / "cache" / "tencent_vectors" / "light_Tencent_AILab_ChineseEmbedding.bin"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    required = set(schema["required"])
    records = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = KeyedVectors.load_word2vec_format(str(VECTOR_PATH), binary=True)
    errors = []
    terms, normalized = set(), set()
    for index, record in enumerate(records, start=1):
        if set(record) != required:
            errors.append(f"record {index}: field mismatch")
            continue
        if record["id"] != index:
            errors.append(f"record {index}: non-consecutive id")
        if record["term"] in terms:
            errors.append(f"record {index}: exact duplicate")
        terms.add(record["term"])
        expected = normalize_term(record["term"])
        if record["normalized_term"] != expected or expected in normalized:
            errors.append(f"record {index}: normalized mismatch/duplicate")
        normalized.add(expected)
        if record["length"] != len(record["term"]) or not 2 <= record["length"] <= 30:
            errors.append(f"record {index}: invalid length")
        if record["review_status"] != "reviewed_keep":
            errors.append(f"record {index}: not reviewed keep")
        if record["term"] not in vectors:
            errors.append(f"record {index}: term absent from Tencent vocabulary")
    if errors:
        print(f"expanded_validation=failed errors={len(errors)}")
        for error in errors[:100]:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"expanded_validation=passed records={len(records)} unique={len(terms)} max_length={max(r['length'] for r in records)}")


if __name__ == "__main__":
    main()
