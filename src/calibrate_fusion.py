"""Calibrate a conservative Tencent+BGE fusion from reviewed samples."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TENCENT_PATH = ROOT / "data" / "final" / "sensitive_words.jsonl"
BGE_PATH = ROOT / "data" / "reranked" / "bge_sensitive_words.jsonl"
OUTPUT_DIR = ROOT / "data" / "reranked"
REPORT_PATH = ROOT / "reports" / "fusion_calibration.json"
LABELED_FILES = (
    ROOT / "reports" / "manual_review_samples_labeled.csv",
    ROOT / "reports" / "bge_manual_review_samples_labeled.csv",
)
FEATURE_NAMES = [
    "tencent_score", "bge_score", "tencent_rank_percentile",
    "bge_rank_percentile", "best_rank_percentile", "worst_rank_percentile",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_matrix(tencent: list[dict], bge_by_term: dict[str, dict]) -> tuple[np.ndarray, list[dict]]:
    total = len(tencent)
    rows, features = [], []
    for record in tencent:
        bge = bge_by_term[record["term"]]
        t_percentile = (record["rank"] - 1) / max(total - 1, 1)
        b_percentile = (bge["bge_rank"] - 1) / max(total - 1, 1)
        features.append([
            float(record["score"]),
            float(bge["bge_score"]),
            t_percentile,
            b_percentile,
            min(t_percentile, b_percentile),
            max(t_percentile, b_percentile),
        ])
        rows.append({**record, **{
            "tencent_score": record["score"],
            "tencent_rank": record["rank"],
            "bge_score": bge["bge_score"],
            "bge_rank": bge["bge_rank"],
            "bge_best_seed": bge["bge_best_seed"],
            "bge_best_category": bge["bge_best_category"],
        }})
    return np.asarray(features, dtype=np.float64), rows


def load_labels() -> dict[str, str]:
    values: dict[str, list[str]] = defaultdict(list)
    for path in LABELED_FILES:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                values[row["term"]].append(row["keep"])
    labels = {}
    for term, decisions in values.items():
        if len(set(decisions)) != 1:
            raise RuntimeError(f"conflicting labels for {term}: {decisions}")
        labels[term] = decisions[0]
    return labels


def main() -> None:
    tencent = load_jsonl(TENCENT_PATH)
    bge = load_jsonl(BGE_PATH)
    bge_by_term = {record["term"]: record for record in bge}
    features, rows = feature_matrix(tencent, bge_by_term)
    labels = load_labels()
    indices = [index for index, row in enumerate(rows) if labels.get(row["term"]) in {"keep", "reject"}]
    x_train = features[indices]
    y_train = np.asarray([labels[rows[index]["term"]] == "keep" for index in indices], dtype=np.int8)

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=20261001),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261001)
    cv_probability = cross_val_predict(estimator, x_train, y_train, cv=folds, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, cv_probability)
    eligible = [index for index, value in enumerate(precision[:-1]) if value >= 0.90]
    threshold_index = max(eligible, key=lambda index: recall[index]) if eligible else int(np.argmax(precision[:-1]))
    threshold = float(thresholds[threshold_index])

    estimator.fit(x_train, y_train)
    probability = estimator.predict_proba(features)[:, 1]
    order = np.argsort(-probability, kind="stable")
    output = []
    for fusion_rank, index in enumerate(order, start=1):
        row = rows[int(index)]
        known_label = labels.get(row["term"])
        if known_label == "keep":
            review_status = "manual_keep"
        elif known_label == "reject":
            review_status = "manual_reject"
        elif known_label == "borderline":
            review_status = "manual_review"
        elif probability[index] >= threshold:
            review_status = "auto_accept_candidate"
        else:
            review_status = "unreviewed"
        output.append({
            "id": fusion_rank,
            "term": row["term"],
            "normalized_term": row["normalized_term"],
            "length": row["length"],
            "source": row["source"],
            "fusion_score": round(float(probability[index]), 8),
            "fusion_rank": fusion_rank,
            "tencent_score": row["tencent_score"],
            "tencent_rank": row["tencent_rank"],
            "bge_score": row["bge_score"],
            "bge_rank": row["bge_rank"],
            "best_seed": row["bge_best_seed"],
            "best_category": row["bge_best_category"],
            "review_status": review_status,
        })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "fused_sensitive_words.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "fused_sensitive_words.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    high_score_terms = {rows[index]["term"] for index in np.flatnonzero(probability >= threshold)}
    auto_accept_unreviewed = sum(term not in labels for term in high_score_terms)
    manual_keep = sum(label == "keep" for label in labels.values())
    report = {
        "method": "standardized logistic fusion; borderline labels excluded",
        "features": FEATURE_NAMES,
        "unique_reviewed_terms": len(labels),
        "training_terms": len(indices),
        "training_keep": int(y_train.sum()),
        "training_reject": int((1 - y_train).sum()),
        "cv_roc_auc": round(float(roc_auc_score(y_train, cv_probability)), 6),
        "cv_average_precision": round(float(average_precision_score(y_train, cv_probability)), 6),
        "selected_probability_threshold": round(threshold, 8),
        "cv_precision_at_threshold": round(float(precision[threshold_index]), 6),
        "cv_recall_at_threshold": round(float(recall[threshold_index]), 6),
        "high_score_candidate_count": len(high_score_terms),
        "auto_accept_unreviewed_count": auto_accept_unreviewed,
        "manual_keep_count": manual_keep,
        "caution": "Cross-validation is on deliberately rank-stratified audit samples, not an iid gold test set; fusion_score is provisional and requires fresh holdout review.",
    }
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
