"""Apply reviewed labels to the fixed BGE reranking QA samples."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "bge_manual_review_samples.csv"
OUTPUT = ROOT / "reports" / "bge_manual_review_samples_labeled.csv"
SUMMARY = ROOT / "reports" / "bge_manual_review_summary.json"

TOP_REJECT = {65, 70, 75, 91, 92, 96, 100}
TOP_BORDERLINE = {55, 66, 68, 69, 74, 78, 81, 87, 99}

MID_KEEP = {954, 957, 958, 960, 961, 963, 979, 998, 1004, 1014, 1032, 1048, 1049}
MID_BORDERLINE = {
    953, 956, 970, 972, 973, 974, 977, 981, 983, 984, 985, 987, 989,
    992, 1001, 1005, 1008, 1010, 1013, 1018, 1021, 1022, 1025, 1028,
    1031, 1039, 1041, 1042, 1045, 1050,
}

TAIL_KEEP = {4917, 4935, 4942, 4953, 4958, 4983, 4986, 4997}
TAIL_BORDERLINE = {
    4901, 4906, 4911, 4912, 4929, 4938, 4941, 4944, 4948, 4950, 4963,
    4977, 4985, 4989, 4990, 4995,
}

COMMENTS = {
    "keep": "与目标语义直接或稳定相关",
    "borderline": "存在主题联想但依赖上下文，保留人工复核",
    "reject": "泛词、碎片、专名或语义漂移",
}


def decision(center: int, rank: int) -> str:
    reject = TOP_REJECT if center == 1 else set()
    borderline = TOP_BORDERLINE if center == 1 else set()
    keep = set()
    if center == 1:
        if rank in reject:
            return "reject"
        if rank in borderline:
            return "borderline"
        return "keep"
    if center == 1000:
        keep, borderline = MID_KEEP, MID_BORDERLINE
    elif center == 5000:
        keep, borderline = TAIL_KEEP, TAIL_BORDERLINE
    else:
        raise ValueError(f"unexpected sample center: {center}")
    return "keep" if rank in keep else "borderline" if rank in borderline else "reject"


def main() -> None:
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 300:
        raise RuntimeError(f"expected 300 rows, got {len(rows)}")
    counts: dict[int, Counter] = defaultdict(Counter)
    score_ranges: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        center, rank = int(row["sample_center"]), int(row["bge_rank"])
        label = decision(center, rank)
        row["keep"], row["comment"] = label, COMMENTS[label]
        counts[center][label] += 1
        score_ranges[center].append(float(row["bge_score"]))
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"total_labeled": len(rows), "bands": {}}
    for center in sorted(counts):
        count = counts[center]
        total = sum(count.values())
        summary["bands"][str(center)] = {
            "count": total,
            "keep": count["keep"],
            "borderline": count["borderline"],
            "reject": count["reject"],
            "strict_keep_rate": round(count["keep"] / total, 4),
            "keep_or_borderline_rate": round((count["keep"] + count["borderline"]) / total, 4),
            "score_min": min(score_ranges[center]),
            "score_max": max(score_ranges[center]),
        }
    summary["finding"] = "BGE improves Top-100 precision but demotes some stable idioms; use score/rank fusion instead of replacing Tencent order."
    with SUMMARY.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
