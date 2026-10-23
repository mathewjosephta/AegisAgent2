"""Train AegisAgent's XGBoost detector on CIC-IDS2017 flow CSVs.

Example:
  python backend/train_model.py --data-dir path/to/extracted/MachineLearningCSV
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def normalize_label(value: object) -> str:
    label = re.sub(r"\s+", " ", str(value)).strip().upper()
    return "BENIGN" if label == "BENIGN" else "ATTACK"


def load_dataset(data_dir: Path, max_rows: int, seed: int):
    # CIC-IDS2017 downloads are normally CSVs; use CSV extension, not PCAP.
    files = sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.CSV"))
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(
            f"No .csv files found under {data_dir}. Extract CIC-IDS2017 MachineLearningCSV.zip first "
            "and ensure the flow tables have a .csv extension."
        )

    frames = []
    for file in files:
        print(f"Reading {file}")
        try:
            frame = pd.read_csv(file, low_memory=False)
        except Exception as exc:
            print(f"Skipping {file}: could not parse CSV ({exc})")
            continue
        frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
        label_cols = [c for c in frame.columns if c.lower() == "label"]
        if not label_cols:
            print(f"Skipping {file}: no Label column")
            continue
        labels = frame[label_cols[0]].map(normalize_label)
        features = frame.drop(columns=label_cols)
        # Keep feature names unique when source files contain duplicate columns.
        seen = {}
        unique_cols = []
        for col in features.columns:
            seen[col] = seen.get(col, 0) + 1
            unique_cols.append(col if seen[col] == 1 else f"{col}.{seen[col]}")
        features.columns = unique_cols
        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna(axis=1, how="all").fillna(0)
        features["__target__"] = labels.to_numpy()
        frames.append(features)

    if not frames:
        raise ValueError("No usable labeled CSVs found. Check that the files are CIC-IDS2017 flow tables.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.replace([np.inf, -np.inf], np.nan)
    y = data.pop("__target__")
    data = data.fillna(0).astype(np.float32)

    if y.nunique() < 2:
        raise ValueError("Training requires both BENIGN and ATTACK rows; the loaded data has only one class.")
    if max_rows > 0 and len(data) > max_rows:
        data, _, y, _ = train_test_split(
            data, y, train_size=max_rows, random_state=seed, stratify=y
        )
        print(f"Stratified sample: {len(data):,} rows")
    return data, y


def main():
    parser = argparse.ArgumentParser(description="Train AegisAgent on CIC-IDS2017 flow data using XGBoost")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing extracted CIC-IDS2017 flow CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/models"))
    parser.add_argument("--max-rows", type=int, default=500_000,
                        help="Maximum stratified rows; use 0 for all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    model = XGBClassifier(
        n_estimators=args.estimators,
        max_depth=args.max_depth,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=args.seed,
    )
    print(f"Training XGBoost on {len(X_train):,} rows with {X_train.shape[1]} features...")
    model.fit(X_train, y_train.map({"BENIGN": 0, "ATTACK": 1}))
    predictions_numeric = model.predict(X_test).astype(int)
    predictions = np.where(predictions_numeric == 1, "ATTACK", "BENIGN")
    y_test_text = y_test.to_numpy()

    report = classification_report(
        y_test_text, predictions, labels=["BENIGN", "ATTACK"],
        output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test_text, predictions, labels=["BENIGN", "ATTACK"])
    metrics = {
        "dataset": "CIC-IDS2017",
        "task": "binary flow classification (BENIGN vs ATTACK)",
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "features": list(X.columns),
        "accuracy": float(accuracy_score(y_test_text, predictions)),
        "precision_attack": float(precision_score(y_test_text, predictions, pos_label="ATTACK", zero_division=0)),
        "recall_attack": float(recall_score(y_test_text, predictions, pos_label="ATTACK", zero_division=0)),
        "f1_attack": float(f1_score(y_test_text, predictions, pos_label="ATTACK", zero_division=0)),
        "confusion_matrix_labels": ["BENIGN", "ATTACK"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "random_state": args.seed,
        "model": "XGBClassifier",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "aegis_xgb.joblib"
    metrics_path = args.output_dir / "metrics.json"
    joblib.dump({"model": model, "features": list(X.columns),
                 "labels": ["BENIGN", "ATTACK"]}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "precision_attack": metrics["precision_attack"],
        "recall_attack": metrics["recall_attack"],
        "f1_attack": metrics["f1_attack"],
        "confusion_matrix": metrics["confusion_matrix"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }, indent=2))
    print("\nPer-class report:")
    print(classification_report(y_test_text, predictions, zero_division=0))


if __name__ == "__main__":
    main()
