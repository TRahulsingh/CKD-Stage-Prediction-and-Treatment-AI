import pandas as pd
import numpy as np
import joblib
import subprocess
import os
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

print("[INFO] Loading models and dataset...")

# Load trained objects
model = joblib.load("models/ckd_stack_model.joblib")
scaler = joblib.load("models/scaler.joblib")
imputer = joblib.load("models/imputer.joblib")
encoder = joblib.load("models/encoder.joblib")
selected_features = joblib.load("models/selected_features.joblib")

# Load dataset
df = pd.read_csv("dataset/kidney_disease_dataset.csv")

# Ensure 'Target' exists for imputer (even if dummy)
if "Target" not in df.columns:
    df["Target"] = np.nan

# Encode categorical columns
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Target']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))
if df["Target"].dtype == "object":
    df["Target"] = encoder.fit_transform(df["Target"].astype(str))

# Feature engineering
if 'Serum creatinine (mg/dl)' in df.columns and 'Hemoglobin level (gms)' in df.columns:
    df['Creatinine_Hemoglobin'] = df['Serum creatinine (mg/dl)'] * df['Hemoglobin level (gms)']
if 'Blood urea (mg/dl)' in df.columns and 'Serum creatinine (mg/dl)' in df.columns:
    df['BUN_Creatinine_Ratio'] = df['Blood urea (mg/dl)'] / (df['Serum creatinine (mg/dl)'] + 1e-5)
if 'Estimated Glomerular Filtration Rate (eGFR)' in df.columns and 'Age of the patient' in df.columns:
    df['eGFR_by_Age'] = df['Estimated Glomerular Filtration Rate (eGFR)'] / (df['Age of the patient'] + 1e-5)

# Log transform
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].min() > 0:
        df[col] = np.log1p(df[col])

# Impute missing values
df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)

# Drop 'Target' AFTER imputation
df_imputed.drop("Target", axis=1, inplace=True)

# Scale features
df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

# Feature selection
X_selected = df_scaled[selected_features]
print(f"[INFO] Input shape after selector: {X_selected.shape}")

# Save for explainers
os.makedirs("shap_plots", exist_ok=True)
X_selected.to_csv("shap_plots/X_sample.csv", index=False)

# Launch SHAP explainers
print("\n[INFO] Launching SHAP explainer subprocesses...\n")

for name in model.named_estimators_.keys():
    print(f"[INFO] Explaining model: {name}...")
    result = subprocess.run(["python", "explain_single.py", name], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"[ERROR] Error for '{name}':\n{result.stderr.strip()}")

print("\n[INFO] SHAP summary plots saved (check shap_plots/ folder).")
