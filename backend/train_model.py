"""Train AegisAgent's Random Forest detector on CIC-IDS2017 flow CSVs.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split


def normalize_label(value: object) -> str:
    label = re.sub(r"\s+", " ", str(value)).strip().upper()
    return "BENIGN" if label == "BENIGN" else "ATTACK"


def load_dataset(data_dir: Path, max_rows: int, seed: int):
    files = sorted(data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir}. Extract CIC-IDS2017 MachineLearningCSV.zip first."
        )

    frames = []
    for file in files:
        print(f"Reading {file}")
        frame = pd.read_csv(file, low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        label_cols = [c for c in frame.columns if c.lower() == "label"]
        if not label_cols:
            print(f"Skipping {file}: no Label column")
            continue
        labels = frame[label_cols[0]].map(normalize_label)
        features = frame.drop(columns=label_cols)
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
        raise ValueError("No usable labeled CSVs found.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data.replace([np.inf, -np.inf], np.nan)
    y = data.pop("__target__")
    data = data.fillna(0).astype(np.float32)

    if max_rows > 0 and len(data) > max_rows:
        data, _, y, _ = train_test_split(
            data, y, train_size=max_rows, random_state=seed, stratify=y
        )
        print(f"Stratified sample: {len(data):,} rows")
    return data, y


def main():
    parser = argparse.ArgumentParser(description="Train AegisAgent on CIC-IDS2017 flow data")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing extracted CIC-IDS2017 CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/models"))
    parser.add_argument("--max-rows", type=int, default=500_000,
                        help="Maximum stratified rows; use 0 for all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    X, y = load_dataset(args.data_dir, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
        min_samples_leaf=2,
    )
    print(f"Training on {len(X_train):,} rows with {X_train.shape[1]} features...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test, predictions, labels=["BENIGN", "ATTACK"],
        output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=["BENIGN", "ATTACK"])
    metrics = {
        "dataset": "CIC-IDS2017",
        "task": "binary flow classification (BENIGN vs ATTACK)",
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "features": list(X.columns),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_attack": float(precision_score(y_test, predictions, pos_label="ATTACK", zero_division=0)),
        "recall_attack": float(recall_score(y_test, predictions, pos_label="ATTACK", zero_division=0)),
        "f1_attack": float(f1_score(y_test, predictions, pos_label="ATTACK", zero_division=0)),
        "confusion_matrix_labels": ["BENIGN", "ATTACK"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "random_state": args.seed,
        "model": "RandomForestClassifier",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "aegis_rf.joblib"
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
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
