"""
Train and evaluate an ML-IDS on NSL-KDD.

Trains two complementary models:
  1. Random Forest  — supervised binary classifier (normal vs. attack)
  2. Isolation Forest — unsupervised anomaly detector (flags novel/unknown
     patterns that don't match the training distribution of 'normal' traffic)

Run:
    python train.py
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)

import pandas as pd
from data_utils import load_dataset

DATA_DIR = "data"
MODEL_DIR = "models"
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")


def main():
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
        raise FileNotFoundError(
            f"Expected NSL-KDD files at {TRAIN_PATH} and {TEST_PATH}. "
            "See README.md for the download link."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    data = load_dataset(TRAIN_PATH, TEST_PATH)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # ---------- Model 1: Random Forest (supervised) ----------
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",   # attacks/normal may be imbalanced
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("\n--- Random Forest Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1 (attack class): {f1_score(y_test, y_pred):.4f}")

    # Feature importance — useful for interpretability / SOC trust
    importances = sorted(
        zip(data["feature_cols"], rf.feature_importances_),
        key=lambda x: -x[1],
    )[:10]
    print("\nTop 10 most important features:")
    for name, score in importances:
        print(f"  {name}: {score:.4f}")

    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))

    # ---------- Per-attack-type breakdown ----------
    # Overall recall hides *which* attacks are being missed. Break it down
    # by the original attack name (before we collapsed to binary) to see
    # whether failures cluster in specific categories (e.g. novel attacks
    # not present in the training set) or are spread evenly.
    print("\n--- Detection rate by attack type (test set) ---")
    breakdown = pd.DataFrame({
        "true_label": data["y_test_multiclass"],
        "predicted_attack": y_pred,
    })
    summary = (
        breakdown.groupby("true_label")
        .agg(count=("predicted_attack", "size"), detected=("predicted_attack", "sum"))
    )
    summary["detection_rate"] = (summary["detected"] / summary["count"]).round(3)
    summary = summary.sort_values("count", ascending=False)
    print(summary.to_string())

    # Flag attack types NSL-KDD's test set includes that never appear in
    # training at all -- these are the hardest cases almost by definition.
    train_attack_types = set(data["y_train_multiclass"])
    unseen_types = [t for t in summary.index if t != "normal" and t not in train_attack_types]
    if unseen_types:
        print(f"\nAttack types in test set NOT seen during training: {unseen_types}")

    # ---------- Model 2: Isolation Forest (unsupervised, novelty detection) ----------
    print("\nTraining Isolation Forest (trained only on normal traffic)...")
    normal_mask = y_train == 0
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train[normal_mask])

    # IsolationForest: -1 = anomaly (flag as attack), 1 = normal
    iso_pred_raw = iso.predict(X_test)
    iso_pred = (iso_pred_raw == -1).astype(int)

    print("\n--- Isolation Forest Evaluation (anomaly = attack) ---")
    print(classification_report(y_test, iso_pred, target_names=["normal", "attack"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, iso_pred))

    joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.joblib"))

    # ---------- Combined: RF OR Isolation Forest ----------
    # RF is strong on attack types it has seen labeled examples of.
    # Isolation Forest doesn't need labels -- it flags anything that
    # deviates from normal traffic, so it can catch attack types RF
    # has never seen. Combining them (flag as attack if EITHER says so)
    # trades some extra false positives on normal traffic for better
    # coverage of novel attacks.
    combined_pred = ((y_pred == 1) | (iso_pred == 1)).astype(int)

    print("\n--- Combined (RF OR Isolation Forest) Evaluation ---")
    print(classification_report(y_test, combined_pred, target_names=["normal", "attack"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, combined_pred))
    print(f"F1 (attack class): {f1_score(y_test, combined_pred):.4f}")

    print("\n--- Combined: detection rate by attack type (test set) ---")
    combined_breakdown = pd.DataFrame({
        "true_label": data["y_test_multiclass"],
        "predicted_attack": combined_pred,
    })
    combined_summary = (
        combined_breakdown.groupby("true_label")
        .agg(count=("predicted_attack", "size"), detected=("predicted_attack", "sum"))
    )
    combined_summary["detection_rate"] = (combined_summary["detected"] / combined_summary["count"]).round(3)
    combined_summary = combined_summary.sort_values("count", ascending=False)
    print(combined_summary.to_string())

    # Save preprocessing objects so predict.py can reproduce them exactly
    joblib.dump(data["encoders"], os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(data["scaler"], os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(data["feature_cols"], os.path.join(MODEL_DIR, "feature_cols.joblib"))

    print(f"\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
