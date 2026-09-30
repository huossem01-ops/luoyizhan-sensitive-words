"""Audit exact, normalized, and highly similar phrases in final retrieval data."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from normalize import char_ngrams, normalize_term


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "final" / "sensitive_words.jsonl"
OUTPUT = ROOT / "reports" / "duplicates_report.csv"


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def edit_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    return 1.0 if not denominator else 1 - levenshtein(left, right) / denominator


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main() -> None:
    with INPUT.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    reports = []
    term_groups = defaultdict(list)
    normalized_groups = defaultdict(list)
    for record in records:
        term_groups[record["term"]].append(record)
        normalized_groups[normalize_term(record["term"])].append(record)

    reported_pairs = set()
    for match_type, groups in (("exact", term_groups), ("normalized", normalized_groups)):
        for values in groups.values():
            for index, left in enumerate(values):
                for right in values[index + 1:]:
                    pair = (left["id"], right["id"])
                    if pair in reported_pairs:
                        continue
                    reported_pairs.add(pair)
                    reports.append((match_type, left, right, 1.0, 1.0))

    grams = {record["id"]: char_ngrams(record["term"], 2) for record in records}
    frequencies = Counter(gram for values in grams.values() for gram in values)
    inverted = defaultdict(list)
    candidates = set()
    by_id = {record["id"]: record for record in records}
    for record in records:
        record_id = record["id"]
        rarest = sorted(grams[record_id], key=lambda gram: (frequencies[gram], gram))[:4]
        for gram in rarest:
            if frequencies[gram] > 250:
                continue
            for other_id in inverted[gram]:
                if abs(len(record["term"]) - len(by_id[other_id]["term"])) <= 3:
                    candidates.add((other_id, record_id))
            inverted[gram].append(record_id)

    for left_id, right_id in sorted(candidates):
        if (left_id, right_id) in reported_pairs:
            continue
        left, right = by_id[left_id], by_id[right_id]
        edit = edit_similarity(normalize_term(left["term"]), normalize_term(right["term"]))
        jac = jaccard(grams[left_id], grams[right_id])
        if edit >= 0.86 or jac >= 0.80:
            reports.append(("fuzzy", left, right, edit, jac))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["match_type", "id_a", "term_a", "id_b", "term_b", "edit_similarity", "jaccard_similarity", "review"]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for match_type, left, right, edit, jac in reports:
            writer.writerow({
                "match_type": match_type,
                "id_a": left["id"], "term_a": left["term"],
                "id_b": right["id"], "term_b": right["term"],
                "edit_similarity": f"{edit:.4f}",
                "jaccard_similarity": f"{jac:.4f}",
                "review": "",
            })
    print(f"records={len(records)} duplicate_candidates={len(reports)}")


if __name__ == "__main__":
    main()
