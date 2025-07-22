# explain_single.py

import sys
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

from catboost import Pool

warnings.filterwarnings("ignore")

# Get model name from CLI
model_name = sys.argv[1]
print(f"[INFO] Explaining model: {model_name}")

try:
    # Load ensemble stack model and get the specific estimator
    model = joblib.load("models/ckd_stack_model.joblib")
    base_model = model.named_estimators_[model_name]

    # Load sample input for SHAP
    X_sample = pd.read_csv("shap_plots/X_sample.csv")

    # Handle model-specific SHAP logic
    if model_name == "cat":
        try:
            print("[INFO] SHAP TreeExplainer initializing for CatBoost...")
            explainer = shap.Explainer(base_model.predict, X_sample)
            print("[INFO] SHAP values computing for CatBoost...")
            shap_values = explainer(X_sample)
        except Exception as ce:
            raise RuntimeError(f"CatBoost SHAP failed: {ce}")

    else:
        print(f"[INFO] TreeExplainer initializing for {model_name.upper()}...")
        explainer = shap.TreeExplainer(base_model)
        print("[INFO] SHAP values computing...")
        shap_values = explainer.shap_values(X_sample)

    # Generate SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary - {model_name.upper()}")
    output_path = f"shap_plots/shap_summary_{model_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"[OK] Saved: {output_path}")

except Exception as e:
    print(f"[ERROR] SHAP failed for model '{model_name}': {e}")
