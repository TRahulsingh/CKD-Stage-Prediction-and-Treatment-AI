# extract_selector.py — Extract selected features from your trained stack model
import joblib
import pandas as pd

print("📦 Loading scaler and imputer...")
scaler = joblib.load("models/scaler.joblib")
imputer = joblib.load("models/imputer.joblib")
encoder = joblib.load("models/encoder.joblib")

print("📄 Loading original dataset...")
df = pd.read_csv("dataset/kidney_disease_dataset.csv")

# Encode like training
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Target']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))
if df['Target'].dtype == 'object':
    df['Target'] = encoder.fit_transform(df['Target'].astype(str))

# Feature engineering
if 'Serum creatinine (mg/dl)' in df.columns and 'Hemoglobin level (gms)' in df.columns:
    df['Creatinine_Hemoglobin'] = df['Serum creatinine (mg/dl)'] * df['Hemoglobin level (gms)']
if 'Blood urea (mg/dl)' in df.columns and 'Serum creatinine (mg/dl)' in df.columns:
    df['BUN_Creatinine_Ratio'] = df['Blood urea (mg/dl)'] / (df['Serum creatinine (mg/dl)'] + 1e-5)
if 'Estimated Glomerular Filtration Rate (eGFR)' in df.columns and 'Age of the patient' in df.columns:
    df['eGFR_by_Age'] = df['Estimated Glomerular Filtration Rate (eGFR)'] / (df['Age of the patient'] + 1e-5)

# Log transform
import numpy as np
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].min() > 0:
        df[col] = np.log1p(df[col])

# Impute (don't drop Target yet!)
df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)

# Drop target, scale
X = df_imputed.drop("Target", axis=1)
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

# Now compare model input shape to X_scaled shape
stack_model = joblib.load("models/ckd_stack_model.joblib")
one_base_model = stack_model.named_estimators_['cat']

# Use a SHAP explainer to get working input
print("📊 Testing to find which 30 features were used...")
import shap
explainer = shap.Explainer(one_base_model.predict, X_scaled)
X_selected = X_scaled.iloc[:, :explainer(X_scaled[:1]).shape[1]]

# Save feature names
print(f"✅ Extracted {X_selected.shape[1]} selected features.")
joblib.dump(list(X_selected.columns), "models/selected_features.joblib")
print("💾 Saved to models/selected_features.joblib")
