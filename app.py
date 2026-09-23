
"""
Flask backend for the ML-Based Intrusion Detection System (ML-IDS).

Features:
1. Serves web_demo.html.
2. Analyzes manually entered network-flow features.
3. Evaluates uploaded CSV datasets using:
   - Random Forest
   - Isolation Forest
   - Combined model (RF OR Isolation Forest)
4. Calculates metrics from the same actual labels and predictions.
5. Generates confusion matrices and record counts.
6. Saves prediction results as a downloadable CSV.

Run:
    python app.py

Open:
    http://127.0.0.1:5000
"""

import os
import uuid
import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    send_file,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from werkzeug.utils import secure_filename


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_DIR = "models"
RESULTS_DIR = "results"
WEB_PAGE = "web_demo.html"

os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)

# Maximum upload size: 50 MB
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ---------------------------------------------------------
# NSL-KDD feature definitions
# ---------------------------------------------------------

FEATURE_COLS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

LABEL_NAMES = ["attack_type", "label", "binary_label", "target"]


# ---------------------------------------------------------
# Load trained model artifacts
# ---------------------------------------------------------

def load_artifacts():
    """Load the models and preprocessing artifacts."""

    required_files = {
        "rf": "random_forest.joblib",
        "iso": "isolation_forest.joblib",
        "encoders": "encoders.joblib",
        "scaler": "scaler.joblib",
        "feature_cols": "feature_cols.joblib",
    }

    artifacts = {}

    for key, filename in required_files.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing model artifact: {path}. "
                "Run train.py first to train and save the models."
            )

        artifacts[key] = joblib.load(path)

    return artifacts


artifacts = load_artifacts()

rf = artifacts["rf"]
iso = artifacts["iso"]
encoders = artifacts["encoders"]
scaler = artifacts["scaler"]
feature_cols = list(artifacts["feature_cols"])


# ---------------------------------------------------------
# Default values for the manual flow analyzer
# ---------------------------------------------------------

DEFAULTS = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 300,
    "dst_bytes": 500,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 5,
    "srv_count": 5,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 0.2,
    "diff_srv_rate": 0.05,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 10,
    "dst_host_srv_count": 10,
    "dst_host_same_srv_rate": 0.2,
    "dst_host_diff_srv_rate": 0.05,
    "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}


# ---------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------

def normalize_category(value, column):
    """Normalize categorical values to match NSL-KDD conventions."""

    value = str(value).strip()

    if column == "protocol_type":
        value = value.lower()

    elif column == "service":
        value = value.lower()

    elif column == "flag":
        value = value.upper()

    return value


def encode_categories(df):
    """Encode categorical columns using the saved LabelEncoders."""

    df = df.copy()

    for col, encoder in encoders.items():
        if col not in df.columns:
            raise ValueError(f"Required categorical feature is missing: {col}")

        values = df[col].apply(lambda x: normalize_category(x, col))

        classes = set(encoder.classes_)

        values = values.apply(
            lambda value: value if value in classes else "__unseen__"
        )

        if "__unseen__" not in classes and any(
            value == "__unseen__" for value in values
        ):
            raise ValueError(
                f"The saved encoder for '{col}' does not contain "
                "'__unseen__'. Retrain the model with the current "
                "data_utils.py preprocessing."
            )

        df[col] = encoder.transform(values)

    return df


def preprocess_features(df):
    """Validate, encode, and scale the 41 input features."""

    df = df.copy()

    missing = [col for col in feature_cols if col not in df.columns]

    if missing:
        raise ValueError(
            "CSV is missing required feature columns: "
            + ", ".join(missing)
        )

    df = df[feature_cols].copy()

    df = encode_categories(df)

    for col in feature_cols:
        if col not in CATEGORICAL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.isnull().any().any():
        bad_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values found in feature columns: "
            + ", ".join(bad_cols)
        )

    X = df.astype(float)

    return scaler.transform(X)


def build_feature_vector(payload):
    """Build and preprocess a single manually submitted flow."""

    row = DEFAULTS.copy()
    row.update(payload)

    df = pd.DataFrame([row])

    return preprocess_features(df)


# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------

def predict_all_models(X):
    """Generate predictions from all three model verdicts."""

    rf_probabilities = rf.predict_proba(X)[:, 1]
    rf_predictions = (rf_probabilities >= 0.5).astype(int)

    iso_raw_predictions = iso.predict(X)
    iso_predictions = (iso_raw_predictions == -1).astype(int)

    combined_predictions = (
        (rf_predictions == 1) | (iso_predictions == 1)
    ).astype(int)

    return (
        rf_predictions,
        rf_probabilities,
        iso_predictions,
        combined_predictions,
    )


def calculate_metrics(y_true, y_pred):
    """
    Calculate binary classification metrics.

    Class 0 = Normal
    Class 1 = Attack

    Every metric and confusion matrix is calculated using the
    exact same y_true and y_pred arrays.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(
            float(precision_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )),
            6,
        ),
        "recall": round(
            float(recall_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )),
            6,
        ),
        "f1": round(
            float(f1_score(
                y_true, y_pred, pos_label=1, zero_division=0
            )),
            6,
        ),
        "confusion_matrix": matrix.tolist(),
    }


# ---------------------------------------------------------
# CSV loading and label processing
# ---------------------------------------------------------

def load_uploaded_csv(file_storage):
    """
    Load a CSV with either:
      - named columns, or
      - NSL-KDD raw rows without a header.

    Supported formats:
      41 features
      41 features + label
      41 features + label + difficulty
    """

    try:
        raw = pd.read_csv(file_storage, header=None)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}")

    if raw.empty:
        raise ValueError("The uploaded CSV file is empty.")

    first_row = raw.iloc[0].astype(str).str.strip().str.lower().tolist()

    recognized_headers = set(FEATURE_COLS) | set(LABEL_NAMES) | {"difficulty"}

    # Determine whether the first row is a header.
    has_header = (
        len(first_row) > 0
        and sum(value in recognized_headers for value in first_row) >= 5
    )

    if has_header:
        file_storage.seek(0)
        df = pd.read_csv(file_storage)
        df.columns = [
            str(col).strip().lower()
            for col in df.columns
        ]
    else:
        column_count = raw.shape[1]

        if column_count == 41:
            raw.columns = FEATURE_COLS

        elif column_count == 42:
            raw.columns = FEATURE_COLS + ["label"]

        elif column_count == 43:
            raw.columns = FEATURE_COLS + ["label", "difficulty"]

        else:
            raise ValueError(
                f"Unexpected CSV column count: {column_count}. "
                "Expected 41 features, 42 columns with a label, "
                "or 43 columns with label and difficulty."
            )

        df = raw.copy()

    # Remove accidental whitespace from string values.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Remove optional NSL-KDD difficulty column.
    df = df.drop(columns=["difficulty"], errors="ignore")

    return df


def extract_labels(df):
    """
    Extract binary ground-truth labels.

    Accepted label columns:
      attack_type, label, binary_label, target

    Text labels:
      normal -> 0
      every other attack name -> 1

    Numeric binary labels:
      0 -> normal
      1 -> attack
    """

    label_col = None

    for candidate in LABEL_NAMES:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        return None, df

    raw_labels = df[label_col].astype(str).str.strip().str.lower()

    # Support labels already encoded as 0 and 1.
    unique_labels = set(raw_labels.unique())

    if unique_labels.issubset({"0", "1"}):
        y_true = raw_labels.astype(int).to_numpy()

    else:
        y_true = (raw_labels != "normal").astype(int).to_numpy()

    features_df = df.drop(columns=[label_col])

    return y_true, features_df


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/")
def home():
    """Serve the web interface."""

    if not os.path.isfile(WEB_PAGE):
        return (
            "Error: web_demo.html was not found. "
            "Place it in the same directory as app.py.",
            404,
        )

    return send_from_directory(".", WEB_PAGE)


# ---------------------------------------------------------
# Manual flow analysis
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """Analyze one manually entered flow."""

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Invalid JSON request. Expected a JSON object."
            }), 400

        X = build_feature_vector(payload)

        (
            rf_predictions,
            rf_probabilities,
            iso_predictions,
            combined_predictions,
        ) = predict_all_models(X)

        rf_pred = int(rf_predictions[0])
        iso_pred = int(iso_predictions[0])
        combined_pred = int(combined_predictions[0])

        return jsonify({
            "rf_prediction": "attack" if rf_pred else "normal",
            "rf_confidence": round(float(rf_probabilities[0]), 4),
            "iso_prediction": "attack" if iso_pred else "normal",
            "combined_prediction": (
                "attack" if combined_pred else "normal"
            ),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------
# CSV dataset evaluation
# ---------------------------------------------------------

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """
    Evaluate an uploaded CSV.

    The response contains:
      - number of records analyzed
      - metrics for each model
      - confusion matrices
      - download URL for the prediction CSV
    """

    if "file" not in request.files:
        return jsonify({
            "error": "No file was uploaded. Select a CSV file first."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({"error": "No file selected."}), 400

    safe_name = secure_filename(uploaded_file.filename)

    if not safe_name.lower().endswith(".csv"):
        return jsonify({
            "error": "Invalid file type. Please upload a .csv file."
        }), 400

    try:
        df = load_uploaded_csv(uploaded_file)

        y_true, features_df = extract_labels(df)

        # Require all 41 features before preprocessing.
        missing = [
            col for col in feature_cols
            if col not in features_df.columns
        ]

        if missing:
            return jsonify({
                "error": (
                    "CSV is missing required feature columns: "
                    + ", ".join(missing)
                )
            }), 400

        X = preprocess_features(features_df)

        (
            rf_predictions,
            rf_probabilities,
            iso_predictions,
            combined_predictions,
        ) = predict_all_models(X)

        records_analyzed = int(len(X))

        if records_analyzed == 0:
            return jsonify({"error": "No records found in the CSV."}), 400

        # Create output data using predictions from this evaluation.
        results = features_df.copy()

        results["rf_prediction"] = np.where(
            rf_predictions == 1, "attack", "normal"
        )

        results["rf_attack_confidence"] = np.round(
            rf_probabilities, 4
        )

        results["iso_prediction"] = np.where(
            iso_predictions == 1, "attack", "normal"
        )

        results["combined_prediction"] = np.where(
            combined_predictions == 1, "attack", "normal"
        )

        response_data = {
            "message": "Dataset evaluation completed.",
            "records_analyzed": records_analyzed,
            "labelled": y_true is not None,
            "models": {},
        }

        # Metrics can only be calculated when ground-truth labels exist.
        if y_true is not None:
            if len(y_true) != records_analyzed:
                return jsonify({
                    "error": (
                        "Label count does not match the number "
                        "of evaluated records."
                    )
                }), 400

            rf_metrics = calculate_metrics(y_true, rf_predictions)
            iso_metrics = calculate_metrics(y_true, iso_predictions)
            combined_metrics = calculate_metrics(
                y_true, combined_predictions
            )

            response_data["models"] = {
                "random_forest": rf_metrics,
                "isolation_forest": iso_metrics,
                "combined": combined_metrics,
            }

            # Also provide convenient direct fields for frontends.
            response_data["random_forest"] = rf_metrics
            response_data["isolation_forest"] = iso_metrics
            response_data["combined"] = combined_metrics

            # Include actual labels in the downloaded file.
            results["actual_label"] = np.where(
                y_true == 1, "attack", "normal"
            )

        else:
            response_data["message"] = (
                "Predictions completed. No recognized label column "
                "was found, so performance metrics were not calculated."
            )

        # Save the prediction CSV with a unique filename.
        unique_name = f"mlids_predictions_{uuid.uuid4().hex}.csv"
        output_path = os.path.join(RESULTS_DIR, unique_name)

        results.to_csv(output_path, index=False)

        response_data["download_url"] = (
            f"/download_predictions/{unique_name}"
        )

        return jsonify(response_data)

    except Exception as exc:
        return jsonify({
            "error": f"CSV evaluation failed: {str(exc)}"
        }), 400


# ---------------------------------------------------------
# Download prediction CSV
# ---------------------------------------------------------

@app.route("/download_predictions/<filename>", methods=["GET"])
def download_predictions(filename):
    """Download a generated prediction CSV."""

    safe_name = secure_filename(filename)

    if safe_name != filename or not safe_name.endswith(".csv"):
        return jsonify({"error": "Invalid download filename."}), 400

    file_path = os.path.join(RESULTS_DIR, safe_name)

    if not os.path.isfile(file_path):
        return jsonify({
            "error": "Prediction file not found. Upload and evaluate again."
        }), 404

    return send_file(
        file_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=safe_name,
    )


# ---------------------------------------------------------
# Error handlers
# ---------------------------------------------------------

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "The uploaded file is too large. Maximum size is 50 MB."
    }), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Requested resource was not found."}), 404


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )