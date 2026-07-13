import streamlit as pd_stream
import requests

# Set page configuration
pd_stream.set_page_config(
    page_title="Clinical Readmission Risk Calculator",
    page_icon="🏥",
    layout="wide"
)

# The URL must use the service name 'backend', NOT '127.0.0.1'
API_URL = "https://diabetes-readmission-backend.onrender.com/predict"


pd_stream.title("🏥 Diabetes Patient Readmission Risk Calculator")
pd_stream.markdown(
    "This interface communicates directly with our production XGBoost service to evaluate "
    "30-day readmission risks using calibrated decision thresholds."
)

pd_stream.divider()

# Organize interface columns
col1, col2 = pd_stream.columns([2, 1])

with col1:
    pd_stream.subheader("📋 Patient Clinical Metrics")
    
    # Sub-columns for neat input density
    sub_col1, sub_col2, sub_col3 = pd_stream.columns(3)
    
    with sub_col1:
        gender = pd_stream.selectbox(
            "Gender", 
            options=[("Female", 0.0), ("Male", 1.0)], 
            format_func=lambda x: x[0],
            key="gender_selectbox"
        )[1]
        
        age_num = pd_stream.slider(
            "Patient Age", 
            min_value=0, 
            max_value=100, 
            value=65, 
            step=1, 
            key="age_slider"
        )
        
        time_in_hospital = pd_stream.slider(
            "Days in Hospital", 
            min_value=1, 
            max_value=14, 
            value=4, 
            key="hospital_days_slider"
        )
        number_diagnoses = pd_stream.slider(
            "Number of Diagnoses", min_value=1, max_value=16, value=9,
            key="diagnoses_slider"
        )

    with sub_col2:
        admission_type_id = pd_stream.number_input(
            "Admission Type ID", min_value=1, max_value=8, value=1, 
            key="admission_type_input"
        )
        discharge_disposition_id = pd_stream.number_input(
            "Discharge Disposition ID", min_value=1, max_value=30, value=1, 
            key="discharge_disp_input"
        )
        admission_source_id = pd_stream.number_input(
            "Admission Source ID", min_value=1, max_value=25, value=7, 
            key="admission_source_input"
        )

    with sub_col3:
        num_lab_procedures = pd_stream.number_input(
            "Number of Lab Procedures", min_value=1, max_value=132, value=45, 
            key="lab_procedures_input"
        )
        num_procedures = pd_stream.number_input(
            "Number of Non-Lab Procedures", min_value=0, max_value=6, value=1, 
            key="procedures_input"
        )
        num_medications = pd_stream.number_input(
            "Number of Medications", min_value=1, max_value=81, value=15, 
            key="medications_input"
        )

    pd_stream.markdown("### 📈 Prior Utilization History")
    use_col1, use_col2, use_col3 = pd_stream.columns(3)
    with use_col1:
        number_outpatient = pd_stream.number_input(
            "Outpatient Visits (Past Year)", min_value=0, value=0, 
            key="outpatient_input"
        )
    with use_col2:
        number_emergency = pd_stream.number_input(
            "Emergency Room Visits (Past Year)", min_value=0, value=0, 
            key="emergency_input"
        )
    with use_col3:
        number_inpatient = pd_stream.number_input(
            "Inpatient Admissions (Past Year)", min_value=0, value=1, 
            key="inpatient_input"
        )

    pd_stream.markdown("### 💊 Medications & Diagnosis Clusters")
    med_col1, med_col2, med_col3 = pd_stream.columns(3)
    with med_col1:
        insulin = pd_stream.checkbox("Prescribed Insulin", value=True, key="insulin_check")
        metformin = pd_stream.checkbox("Prescribed Metformin", value=False, key="metformin_check")
    with med_col2:
        diabetesMed = pd_stream.checkbox("Any Diabetes Medication", value=True, key="diabetes_med_check")
        diag_1_group_Diabetes = pd_stream.checkbox("Primary Diagnosis: Diabetes", value=True, key="diag1_check")
    with med_col3:
        diag_2_group_Circulatory = pd_stream.checkbox("Secondary Diagnosis: Circulatory Disease", value=False, key="diag2_check")

with col2:
    pd_stream.subheader("🔮 Pipeline Risk Assessment")
    
    # Construct input payload matching FastAPI Pydantic requirements
    payload = {
        "gender": gender,
        "admission_type_id": int(admission_type_id),
        "discharge_disposition_id": int(discharge_disposition_id),
        "admission_source_id": int(admission_source_id),
        "time_in_hospital": int(time_in_hospital),
        "num_lab_procedures": int(num_lab_procedures),
        "num_procedures": int(num_procedures),
        "num_medications": int(num_medications),
        "number_outpatient": int(number_outpatient),
        "number_emergency": int(number_emergency),
        "number_inpatient": int(number_inpatient),
        "number_diagnoses": int(number_diagnoses),
        "age_num": float(age_num),
        "features": {
            "insulin": int(insulin),
            "metformin": int(metformin),
            "diabetesMed": int(diabetesMed),
            "diag_1_group_Diabetes": int(diag_1_group_Diabetes),
            "diag_2_group_Circulatory": int(diag_2_group_Circulatory)
        }
    }
    
    if pd_stream.button("Run Inference", type="primary", use_container_width=True, key="run_inference_btn"):
        try:
            with pd_stream.spinner("Evaluating models across decision baselines..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5.0)
                
            if response.status_code == 200:
                data = response.json()
                prob = data["readmission_probability"]
                decisions = data["decisions"]
                
                # Visual Metric Display
                pd_stream.metric(label="Calculated Readmission Risk", value=f"{prob * 100:.1f}%")
                
                # Dynamic alert coloring based on standard baseline
                if prob > 0.50:
                    pd_stream.error("⚠️ HIGH RISK: Immediate post-discharge support recommended.")
                elif prob > 0.45:
                    pd_stream.warning("⚠️ ELEVATED RISK: Flagged under aggressive screening thresholds.")
                else:
                    pd_stream.success("✅ LOW RISK: Standard discharge workflow acceptable.")
                
                # Threshold Matrix Breakdown
                pd_stream.markdown("### Operational Settings")
                
                def render_flag(val):
                    return "🔴 Intervention Flagged" if val == 1 else "🟢 Normal Track"

                pd_stream.info(f"**Aggressive Screening (0.45):** {render_flag(decisions['threshold_0_45'])}")
                pd_stream.info(f"**Standard Balanced Baseline (0.50):** {render_flag(decisions['threshold_0_50'])}")
                pd_stream.info(f"**Resource-Constrained Target (0.55):** {render_flag(decisions['threshold_0_55'])}")
                
            else:
                pd_stream.error(f"Prediction Service returned error code: {response.status_code}")
        except Exception as e:
            pd_stream.error(f"Could not reach backend API layer. Is the FastAPI service active? Error: {e}")