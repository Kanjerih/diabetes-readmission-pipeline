import os
import json
import subprocess
from datetime import datetime, timezone
import mlflow
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
    # MLflow: write tracking data to a local ./mlruns folder (no server needed).
    # View the dashboard later with: mlflow ui
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-readmission")

    with mlflow.start_run():
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
        mlflow.log_params(hyperparameters)
        mlflow.log_param("data_path", data_path)

        xgb_model = XGBClassifier(**hyperparameters)

        xgb_model.fit(X_train, y_train)
        print("Training complete!")

        # 4. Evaluate baseline performance metrics
        y_pred_xgb = xgb_model.predict(X_test)
        y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba_xgb)
        mlflow.log_metric("roc_auc", roc_auc)

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

            # Log key metrics per threshold, using mlflow-safe metric names (no dots/spaces issues)
            t_label = str(t).replace(".", "_")
            mlflow.log_metric(f"precision_readmit_t{t_label}", report['Readmitted <30']['precision'])
            mlflow.log_metric(f"recall_readmit_t{t_label}", report['Readmitted <30']['recall'])
            mlflow.log_metric(f"f1_readmit_t{t_label}", report['Readmitted <30']['f1-score'])

        # ==========================================
        # 6. VERSIONED MODEL SAVE + METADATA
        # ==========================================
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        versions_dir = os.path.join(os.path.dirname(model_save_path), "versions")
        os.makedirs(versions_dir, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit_hash = get_git_commit_hash()
        version_id = f"{timestamp}_{commit_hash}"

        mlflow.set_tag("version_id", version_id)
        mlflow.set_tag("git_commit", commit_hash)

        versioned_filename = f"diabetes_readmission_xgb_model_{version_id}.json"
        versioned_path = os.path.join(versions_dir, versioned_filename)

        # Save the versioned, permanent copy -- this always happens, pass or fail,
        # so every training run is recorded even if it doesn't get promoted
        xgb_model.save_model(versioned_path)
        print(f"\nSaved versioned model artifact to: {versioned_path}")
        mlflow.log_artifact(versioned_path, artifact_path="model")

        # ==========================================
        # 6b. VALIDATION GATE: only promote to "latest" if this model
        # doesn't regress meaningfully against the best ROC-AUC on record
        # ==========================================
        registry_path = os.path.join(os.path.dirname(model_save_path), "registry.json")
        previous_registry = []
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                try:
                    previous_registry = json.load(f)
                except json.JSONDecodeError:
                    previous_registry = []

        ROC_AUC_TOLERANCE = 0.01  # allow small noise; anything worse than this blocks promotion
        previous_best_auc = None
        if previous_registry:
            previous_best_auc = max(entry["roc_auc"] for entry in previous_registry)

        passed_validation = True
        if previous_best_auc is not None and roc_auc < (previous_best_auc - ROC_AUC_TOLERANCE):
            passed_validation = False
            print(f"\n VALIDATION GATE FAILED: new ROC-AUC ({roc_auc:.4f}) is worse than "
                  f"the previous best on record ({previous_best_auc:.4f}) by more than "
                  f"the allowed tolerance ({ROC_AUC_TOLERANCE}).")
            print(" This version will be recorded, but NOT promoted to production.")
        else:
            if previous_best_auc is not None:
                print(f"\n Validation gate passed: ROC-AUC {roc_auc:.4f} vs. previous best {previous_best_auc:.4f}")
            else:
                print(f"\n Validation gate passed: no previous version on record, {roc_auc:.4f} is the new baseline")

        mlflow.set_tag("promoted_to_production", str(passed_validation))
        mlflow.log_metric("passed_validation_gate", 1 if passed_validation else 0)

        json_path = model_save_path.replace(".pkl", ".json")
        if passed_validation:
            # Overwrite the fixed-path "latest" model that the API service loads
            xgb_model.save_model(json_path)
            print(f"Updated production 'latest' model at: {json_path}")
        else:
            print(f"Skipped updating production model at: {json_path} (validation gate blocked promotion)")

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
            "model_file": versioned_filename,
            "promoted_to_production": passed_validation
        }
        metadata_path = os.path.join(versions_dir, f"diabetes_readmission_xgb_model_{version_id}.metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_path}")
        mlflow.log_artifact(metadata_path, artifact_path="metadata")

        # Append to the running registry log (reusing the registry already loaded for the gate check above)
        previous_registry.append({
            "version_id": version_id,
            "timestamp_utc": timestamp,
            "git_commit": commit_hash,
            "roc_auc": roc_auc,
            "model_file": versioned_filename,
            "metadata_file": os.path.basename(metadata_path),
            "promoted_to_production": passed_validation
        })

        with open(registry_path, "w") as f:
            json.dump(previous_registry, f, indent=2)
        print(f"Updated model registry at: {registry_path}")
        print(f"\nModel version: {version_id}")

        if not passed_validation:
            print("\nExiting with error status: validation gate blocked this model from production.")
            raise SystemExit(1)


if __name__ == "__main__":
    train_pipeline("data/diabetic_data.csv", "models/diabetes_readmission_xgb_model.json")
