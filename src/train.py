import os
import json
import subprocess
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from src.preprocess import pipeline_preprocess


def get_git_commit_hash():
    """Best-effort retrieval of the current git commit hash, for traceability.
    Returns 'unknown' if not in a git repo or git isn't available (e.g. some CI contexts)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def train_pipeline(data_path, model_save_path):
    print(" Ingesting and preprocessing dataset...")
    X, y = pipeline_preprocess(data_path)

    # 1. Split data cleanly matching structural validation splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Calculate the exact scale_pos_weight for XGBoost
    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    scale_weight_value = num_negative / num_positive
    print(f"Calculated XGBoost Class Weight multiplier: {scale_weight_value:.2f}")

    # 3. Initialize XGBoost with optimized hyperparameters
    print("\nTraining XGBoost model...")
    hyperparameters = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_weight_value,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1
    }
    xgb_model = XGBClassifier(**hyperparameters)

    xgb_model.fit(X_train, y_train)
    print("Training complete!")

    # 4. Evaluate baseline performance metrics
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba_xgb)

    print("\n================ XGBOOST CLINICAL PERFORMANCE ================")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=['No Readmit / >30 Days', 'Readmitted <30 Days']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # 5. Evaluate custom operational thresholds
    threshold_reports = {}
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    for t in thresholds:
        y_pred_custom = (y_proba_xgb > t).astype(int)
        report = classification_report(
            y_test, y_pred_custom,
            target_names=['No Readmit', 'Readmitted <30'],
            output_dict=True
        )
        threshold_reports[str(t)] = report
        print(f"\n========== THRESHOLD: {t} ==========")
        print(classification_report(y_test, y_pred_custom, target_names=['No Readmit', 'Readmitted <30']))

    # ==========================================
    # 6. VERSIONED MODEL SAVE + METADATA
    # ==========================================
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    versions_dir = os.path.join(os.path.dirname(model_save_path), "versions")
    os.makedirs(versions_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    commit_hash = get_git_commit_hash()
    version_id = f"{timestamp}_{commit_hash}"

    versioned_filename = f"diabetes_readmission_xgb_model_{version_id}.json"
    versioned_path = os.path.join(versions_dir, versioned_filename)

    # Save the versioned, permanent copy
    xgb_model.save_model(versioned_path)
    print(f"\nSaved versioned model artifact to: {versioned_path}")

    # Also overwrite the fixed-path "latest" model that the API service loads
    json_path = model_save_path.replace(".pkl", ".json")
    xgb_model.save_model(json_path)
    print(f"Updated production 'latest' model at: {json_path}")

    # Write metadata for this version
    metadata = {
        "version_id": version_id,
        "timestamp_utc": timestamp,
        "git_commit": commit_hash,
        "data_path": data_path,
        "hyperparameters": hyperparameters,
        "metrics": {
            "roc_auc": roc_auc,
            "threshold_reports": threshold_reports
        },
        "num_features": X.shape[1],
        "num_train_rows": len(X_train),
        "num_test_rows": len(X_test),
        "model_file": versioned_filename
    }
    metadata_path = os.path.join(versions_dir, f"diabetes_readmission_xgb_model_{version_id}.metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # Append to the running registry log
    registry_path = os.path.join(os.path.dirname(model_save_path), "registry.json")
    registry = []
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            try:
                registry = json.load(f)
            except json.JSONDecodeError:
                registry = []

    registry.append({
        "version_id": version_id,
        "timestamp_utc": timestamp,
        "git_commit": commit_hash,
        "roc_auc": roc_auc,
        "model_file": versioned_filename,
        "metadata_file": os.path.basename(metadata_path)
    })

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Updated model registry at: {registry_path}")
    print(f"\nModel version: {version_id}")


if __name__ == "__main__":
    train_pipeline("data/diabetic_data.csv", "models/diabetes_readmission_xgb_model.json")
