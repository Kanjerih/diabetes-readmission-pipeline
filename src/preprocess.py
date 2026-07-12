import pandas as pd
import numpy as np

def pipeline_preprocess(filepath):
    # Ingest data with proper missing token mapping
    data = pd.read_csv(filepath, na_values="?", low_memory=False)
    
    # Columns with extremely high levels of missing data (typically above 80%) were dropped
    data.drop(columns=['weight', 'max_glu_serum', 'A1Cresult'], inplace=True, errors='ignore')
    
    # Missing values were consistently encoded using the label 'missing' across categorical features
    data['medical_specialty'] = data['medical_specialty'].fillna('missing')
    data['payer_code'] = data['payer_code'].fillna('missing')
    
    # High-cardinality categorical variables such as medical_specialty and payer_code were reduced
    top_payers = data['payer_code'].value_counts().nlargest(5).index
    data['payer_code'] = data['payer_code'].apply(lambda x: x if x in top_payers else 'Other')
    
    top_specs = data['medical_specialty'].value_counts().nlargest(10).index
    data['medical_specialty'] = data['medical_specialty'].apply(lambda x: x if x in top_specs else 'Other')
    
    data['race'] = data['race'].fillna('missing')
    for col in ['diag_1', 'diag_2', 'diag_3']:
        data[col] = data[col].fillna('missing')
    data['race'] = data['race'].replace({'AfricanAmerican': 'African American'})
    
    # The columns 'encounter_id' and 'patient_nbr' were removed because they are purely identifier variables
    data.drop(columns=['encounter_id', 'patient_nbr'], inplace=True, errors='ignore')
    
    # Convert age ranges to numeric midpoints
    def convert_age(age_str):
        age_str = age_str.strip('[)')
        lower, upper = age_str.split('-')
        return (int(lower) + int(upper)) / 2

    data['age_num'] = data['age'].apply(convert_age)
    data = data.drop('age', axis=1, errors='ignore')
    
    # ==========================================
    # 1. MEDICATION & MEDICINE MAPPING
    # ==========================================
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for col in med_cols:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: 0 if x == 'No' else 1)

    data['change'] = data['change'].map({'No': 0, 'Ch': 1})
    data['diabetesMed'] = data['diabetesMed'].map({'No': 0, 'Yes': 1})

    # Drop columns with zero variance before encoding anything else
    data = data.drop(['examide', 'citoglipton'], axis=1, errors='ignore')

    # ==========================================
    # 2. DIAGNOSIS GROUPING & ONE-HOT ENCODING
    # ==========================================
    def map_diag(diag):
        try:
            diag = float(diag)
        except:
            return 'Other'

        if 390 <= diag <= 459 or diag == 785:
            return 'Circulatory'
        elif 460 <= diag <= 519 or diag == 786:
            return 'Respiratory'
        elif 520 <= diag <= 579 or diag == 787:
            return 'Digestive'
        elif diag == 250:
            return 'Diabetes'
        elif 800 <= diag <= 999:
            return 'Injury'
        else:
            return 'Other'

    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in data.columns:
            data[col + '_group'] = data[col].apply(map_diag)

    data = data.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, errors='ignore')

    diag_cols = ['diag_1_group', 'diag_2_group', 'diag_3_group']
    data = pd.get_dummies(data, columns=[c for c in diag_cols if c in data.columns], drop_first=True)

    # ==========================================
    # 3. TARGET & CATEGORICAL FEATURE MAPPING
    # ==========================================
    data['readmitted'] = data['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0}).astype(int)

    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['gender'] = data['gender'].fillna(data['gender'].mode()[0])

    data = pd.get_dummies(data, columns=['race'], drop_first=True)
    data = pd.get_dummies(data, columns=['payer_code'], drop_first=True)
    data = pd.get_dummies(data, columns=['medical_specialty'], drop_first=True)

    # ==========================================
    # 4. CLEANUP DUMMY BASELINES & CONVERT BOOLEANS
    # ==========================================
    cols_to_drop = [
        'diag_1_group_Other', 'diag_2_group_Other', 'diag_3_group_Other',
        'race_missing', 'payer_code_missing', 'medical_specialty_missing',
        'medical_specialty_Other', 'payer_code_Other'
    ]
    data = data.drop(columns=cols_to_drop, errors='ignore')

    # Turn booleans into 1s and 0s cleanly for XGBoost
    bool_cols = data.select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    # ==========================================
    # 5. CUSTOM PREDICTIVE FEATURES INJECTION
    # ==========================================
    # 1. Total Healthcare Utilization Index (Strong indicator of chronic frailty)
    data['total_visits'] = data['number_inpatient'] + data['number_outpatient'] + data['number_emergency']

    # 2. Medication Density (More medications over fewer days indicates acute severity)
    data['meds_per_day'] = data['num_medications'] / (data['time_in_hospital'] + 1)

    # 3. Micro-Comorbidities (Tracks intersections of simultaneous high-risk conditions)
    if 'diag_1_group_Diabetes' in data.columns and 'diag_2_group_Diabetes' in data.columns:
        data['diab_and_circ'] = data['diag_1_group_Diabetes'] * data['diag_2_group_Diabetes']
    else:
        data['diab_and_circ'] = 0

    X = data.drop('readmitted', axis=1)
    y = data['readmitted']
    
    return X, y