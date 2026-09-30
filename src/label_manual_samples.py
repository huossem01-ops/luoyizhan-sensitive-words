"""Apply the first manually reviewed labels to the three fixed QA samples.

This file intentionally stores explicit rank decisions instead of inventing a
classifier: it is a reproducible record of the small LLM-assisted/manual audit.
Unlisted Top-100 rows are kept; unlisted lower-band rows are rejected.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "manual_review_samples.csv"
OUTPUT = ROOT / "reports" / "manual_review_samples_labeled.csv"
SUMMARY = ROOT / "reports" / "manual_review_summary.json"

TOP_REJECT = {49, 51, 54, 62, 64, 65, 68, 69, 81, 87, 90, 94, 98}
TOP_BORDERLINE = {72, 76, 89, 93}

MID_KEEP = {
    951, 953, 958, 961, 968, 970, 974, 977, 978, 980, 982, 983, 984,
    985, 1007, 1014, 1016, 1018, 1029, 1031, 1035, 1041, 1042,
}
MID_BORDERLINE = {
    954, 964, 965, 973, 979, 988, 994, 998, 999, 1001, 1003, 1005,
    1008, 1009, 1010, 1011, 1022, 1027, 1028, 1032, 1033, 1036,
    1040, 1043, 1047, 1048,
}

TAIL_KEEP = {4942, 4981, 4997}
TAIL_BORDERLINE = {
    4907, 4916, 4918, 4919, 4923, 4925, 4929, 4938, 4941, 4946,
    4948, 4949, 4956, 4965, 4971, 4973, 4975, 4976, 4978, 4980,
    4983, 4987, 4994, 4998, 4999,
}


def decision(center: int, rank: int) -> str:
    if center == 1:
        if rank in TOP_REJECT:
            return "reject"
        if rank in TOP_BORDERLINE:
            return "borderline"
        return "keep"
    if center == 1000:
        if rank in MID_KEEP:
            return "keep"
        if rank in MID_BORDERLINE:
            return "borderline"
        return "reject"
    if center == 5000:
        if rank in TAIL_KEEP:
            return "keep"
        if rank in TAIL_BORDERLINE:
            return "borderline"
        return "reject"
    raise ValueError(f"unexpected sample center: {center}")


COMMENTS = {
    "keep": "与恋爱、婚恋、亲密、单身或校园恋爱语义直接或稳定相关",
    "borderline": "存在主题联想但依赖上下文，暂不进入保守自动接收集合",
    "reject": "泛词、上下文碎片、专名或家庭/校园/场景语义漂移",
}


def main() -> None:
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300:
        raise RuntimeError(f"expected 300 review rows, got {len(rows)}")

    counts: dict[int, Counter] = defaultdict(Counter)
    score_ranges: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        center, rank = int(row["sample_center"]), int(row["rank"])
        label = decision(center, rank)
        row["keep"] = label
        row["comment"] = COMMENTS[label]
        counts[center][label] += 1
        score_ranges[center].append(float(row["score"]))

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "annotation_policy": "strict semantic relevance; borderline is excluded from conservative auto-accept precision",
        "total_labeled": len(rows),
        "bands": {},
        "recommended_thresholds": {
            "auto_accept_min_score": 0.74,
            "manual_review_min_score": 0.58,
            "below_review_threshold": "reject_by_default",
            "rationale": "Top-100 remains high precision while the rank-1000 band already has low strict precision; thresholds are provisional until denser samples are labeled.",
        },
    }
    for center in sorted(counts):
        band_counts = counts[center]
        total = sum(band_counts.values())
        summary["bands"][str(center)] = {
            "count": total,
            "keep": band_counts["keep"],
            "borderline": band_counts["borderline"],
            "reject": band_counts["reject"],
            "strict_keep_rate": round(band_counts["keep"] / total, 4),
            "keep_or_borderline_rate": round((band_counts["keep"] + band_counts["borderline"]) / total, 4),
            "score_min": min(score_ranges[center]),
            "score_max": max(score_ranges[center]),
        }
    with SUMMARY.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
