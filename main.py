import os
import sys
from src.train import train_pipeline

def main():
    DATA_PATH = os.path.join("data", "diabetic_data.csv")
    MODEL_PATH = os.path.join("models", "diabetes_readmission_xgb_model.json")
    
    print("====================================================")
    print(" STARTING MACHINE LEARNING LIFECYCLE PIPELINE")
    print("====================================================")
    
    if not os.path.exists(DATA_PATH):
        print(f" Error: Raw dataset file not found at '{DATA_PATH}'.")
        print(" Please copy 'diabetic_data.csv' into the 'data/' directory.")
        sys.exit(1)
        
    try:
        train_pipeline(DATA_PATH, MODEL_PATH)
        print("\n====================================================")
        print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("====================================================")
    except Exception as e:
        print(f"\n Pipeline execution failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()