"""
Score new traffic records with the trained Random Forest IDS model.

Input CSV must have the same 41 NSL-KDD columns (no label column needed).
See data_utils.COLUMN_NAMES for the expected column order/names.

Run:
    python predict.py path/to/new_flows.csv
"""

import sys
import os
import joblib
import pandas as pd

MODEL_DIR = "models"


def load_artifacts():
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    return rf, encoders, scaler, feature_cols


def preprocess(df: pd.DataFrame, encoders, scaler, feature_cols) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        df[col] = df[col].apply(lambda v: v if v in le.classes_ else "__unseen__")
        df[col] = le.transform(df[col])
    X = df[feature_cols].astype(float)
    return scaler.transform(X)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/new_flows.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    rf, encoders, scaler, feature_cols = load_artifacts()

    df = pd.read_csv(input_path)
    X = preprocess(df, encoders, scaler, feature_cols)

    preds = rf.predict(X)
    proba = rf.predict_proba(X)[:, 1]

    results = df.copy()
    results["prediction"] = ["attack" if p == 1 else "normal" for p in preds]
    results["attack_confidence"] = proba.round(4)

    flagged = results[results["prediction"] == "attack"]
    print(f"Scored {len(results)} records — {len(flagged)} flagged as attacks.\n")
    if len(flagged) > 0:
        print(flagged[["prediction", "attack_confidence"]].to_string())

    out_path = input_path.replace(".csv", "_scored.csv")
    results.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
