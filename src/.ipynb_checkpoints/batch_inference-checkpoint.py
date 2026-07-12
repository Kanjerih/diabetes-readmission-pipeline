import asyncio
import time
import pandas as pd
import httpx

# Configuration
API_URL = "http://127.0.0.1:8000/predict"
CONCURRENT_REQUESTS = 20  # Semaphore limit to prevent overwhelming the API

async def score_single_patient(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, record: dict, index: int):
    """Sends a single patient record to the FastAPI endpoint with rate limiting via Semaphore."""
    # Restructure the row flat format into the expected Pydantic layout
    base_features = [
        "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "age_num"
    ]
    
    payload = {k: record[k] for k in base_features if k in record}
    # Package all remaining columns inside the dynamic 'features' sub-dictionary
    payload["features"] = {k: v for k, v in record.items() if k not in base_features}

    async with semaphore:
        try:
            response = await client.post(API_URL, json=payload, timeout=10.0)
            if response.status_code == 200:
                res_json = response.json()
                return {
                    "index": index,
                    "probability": res_json["readmission_probability"],
                    "flag_45": res_json["decisions"]["threshold_0_45"],
                    "flag_50": res_json["decisions"]["threshold_0_50"],
                    "flag_55": res_json["decisions"]["threshold_0_55"],
                    "status": "Success"
                }
            else:
                return {"index": index, "status": f"Error {response.status_code}"}
        except Exception as e:
            return {"index": index, "status": f"Failed: {str(e)}"}

async def process_batch(csv_path: str, output_path: str):
    """Reads input CSV, coordinates async tasks, and appends pipeline scores."""
    print(f"📖 Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    start_time = time.time()
    
    print(f"🚀 Scaling inference pipeline concurrently across {len(records)} patient rows...")
    
    # Maintain an open client connection pool throughout the batch cycle
    async with httpx.AsyncClient() as client:
        tasks = [
            score_single_patient(client, semaphore, record, idx) 
            for idx, record in enumerate(records)
        ]
        results = await asyncio.gather(*tasks)
    
    # Map scored data fields back onto our primary dataframe architecture
    results_df = pd.DataFrame(results).set_index("index")
    
    df["readmission_probability"] = results_df["probability"]
    df["flag_threshold_0_45"] = results_df["flag_45"]
    df["flag_threshold_0_50"] = results_df["flag_50"]
    df["flag_threshold_0_55"] = results_df["flag_55"]
    df["inference_status"] = results_df["status"]
    
    # Save results
    df.to_csv(output_path, index=False)
    elapsed = time.time() - start_time
    print(f"✅ Batch complete! Processed {len(records)} rows in {elapsed:.2f} seconds.")
    print(f"💾 Results saved directly to: {output_path}")

if __name__ == "__main__":
    # Example placeholder paths—adjust to match your real data directories
    input_csv = "data/test_patients.csv" 
    output_csv = "data/batch_predictions_output.csv"
    
    import os
    if os.path.exists(input_csv):
        asyncio.run(process_batch(input_csv, output_csv))
    else:
        print(f"⚠️ Verification Error: Could not find target mock file at '{input_csv}' to test.")