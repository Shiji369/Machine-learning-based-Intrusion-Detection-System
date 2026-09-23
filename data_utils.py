"""
Loading and preprocessing for the NSL-KDD intrusion detection dataset.

NSL-KDD has 41 features + 1 label column (+ a difficulty score column
in some releases, which we drop). Features are a mix of:
  - 3 categorical: protocol_type, service, flag
  - 38 numeric: duration, byte counts, error rates, etc.

Labels are specific attack names (e.g. 'neptune', 'smurf', 'normal').
We collapse them into a binary label (normal=0 / attack=1) for the
main classifier, but keep the original multi-class label available too.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw NSL-KDD .txt file (comma-separated, no header)."""
    df = pd.read_csv(path, names=COLUMN_NAMES)
    df = df.drop(columns=["difficulty"], errors="ignore")
    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the specific attack name into normal=0 / attack=1."""
    df = df.copy()
    df["binary_label"] = (df["label"] != "normal").astype(int)
    return df


def encode_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit encoders/scaler on train, apply to both train and test.
    Returns (X_train, X_test, encoders, scaler, feature_names).
    Unseen categories in test are mapped to a fallback 'unknown' bucket.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(list(train_df[col].unique()) + ["__unseen__"])
        train_df[col] = le.transform(train_df[col])

        test_df[col] = test_df[col].apply(
            lambda v: v if v in le.classes_ else "__unseen__"
        )
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le

    feature_cols = [c for c in COLUMN_NAMES if c not in ("label", "difficulty")]

    X_train = train_df[feature_cols].astype(float)
    X_test = test_df[feature_cols].astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, encoders, scaler, feature_cols


def load_dataset(train_path: str, test_path: str):
    """Convenience wrapper: load, label, encode, scale. Returns a dict of arrays."""
    train_df = add_binary_label(load_raw(train_path))
    test_df = add_binary_label(load_raw(test_path))

    X_train, X_test, encoders, scaler, feature_cols = encode_and_scale(train_df, test_df)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": train_df["binary_label"].values,
        "y_test": test_df["binary_label"].values,
        "y_train_multiclass": train_df["label"].values,
        "y_test_multiclass": test_df["label"].values,
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
