import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from src.preprocess import pipeline_preprocess

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
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_weight_value,  # Forces XGBoost to focus heavily on class 1
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    print("Training complete!")
    
    # 4. Evaluate baseline performance metrics
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    print("\n================ XGBOOST CLINICAL PERFORMANCE ================")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba_xgb):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=['No Readmit / >30 Days', 'Readmitted <30 Days']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))
    
    # 5. Evaluate custom operational thresholds
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    for t in thresholds:
        y_pred_custom = (y_proba_xgb > t).astype(int)
        print(f"\n========== THRESHOLD: {t} ==========")
        print(classification_report(y_test, y_pred_custom, target_names=['No Readmit', 'Readmitted <30']))
        
    # Save the new production-grade model artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # Ensure your path ends in .json
    json_path = model_save_path.replace(".pkl", ".json")
    xgb_model.save_model(json_path)
    print(f"\nSuccessfully saved optimized XGBoost model to: {json_path}")

if __name__ == "__main__":
    # Update the filename here as well
    train_pipeline("data/diabetic_data.csv", "models/diabetes_readmission_xgb_model.json")