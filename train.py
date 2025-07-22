# train.py — Ultimate CKD Stage Prediction Pipeline with Meta-Stacking and Advanced Feature Selection

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, StackingClassifier
from imblearn.combine import SMOTETomek

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna
import joblib
import os

print("\U0001F4C5 Loading dataset...")
df = pd.read_csv("dataset/kidney_disease_dataset.csv")

# Encode categorical variables
le = LabelEncoder()
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Target']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))
if df['Target'].dtype == 'object':
    df['Target'] = le.fit_transform(df['Target'].astype(str))

# Feature Engineering
if 'Serum creatinine (mg/dl)' in df.columns and 'Hemoglobin level (gms)' in df.columns:
    df['Creatinine_Hemoglobin'] = df['Serum creatinine (mg/dl)'] * df['Hemoglobin level (gms)']
if 'Blood urea (mg/dl)' in df.columns and 'Serum creatinine (mg/dl)' in df.columns:
    df['BUN_Creatinine_Ratio'] = df['Blood urea (mg/dl)'] / (df['Serum creatinine (mg/dl)'] + 1e-5)
if 'Estimated Glomerular Filtration Rate (eGFR)' in df.columns and 'Age of the patient' in df.columns:
    df['eGFR_by_Age'] = df['Estimated Glomerular Filtration Rate (eGFR)'] / (df['Age of the patient'] + 1e-5)

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].min() > 0:
        df[col] = np.log1p(df[col])

# Imputation
print("\U0001F50D Imputing missing values...")
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Scaling and splitting features
scaler = StandardScaler()
X = df_imputed.drop('Target', axis=1)
y = df_imputed['Target']
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Remove outliers
print("\u274c Removing outliers with IsolationForest...")
iso = IsolationForest(contamination=0.02, random_state=42)
outliers = iso.fit_predict(X_scaled)
mask = outliers != -1
X_cleaned, y_cleaned = X_scaled[mask], y[mask]

# Feature Selection (Mutual Information)
print("🔍 Selecting best features...")
selector = SelectKBest(score_func=mutual_info_classif, k=30)
X_selected = selector.fit_transform(X_cleaned, y_cleaned)
selected_features = X_cleaned.columns[selector.get_support()]
X_final = pd.DataFrame(X_selected, columns=selected_features)

# Resampling
print("⚖️ Balancing classes with SMOTETomek...")
smote = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_final, y_cleaned)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Optimize CatBoost with Optuna
print("🧠 Tuning CatBoost with Optuna...")
def objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 6, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
        'iterations': trial.suggest_int('iterations', 300, 600),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 5),
        'random_state': 42,
        'verbose': 0,
        'task_type': 'CPU',
        'early_stopping_rounds': 50
    }
    
    X_np = X_train.values
    y_np = y_train.values
    scores = []
    for train_idx, val_idx in StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X_np, y_np):
        model = CatBoostClassifier(**params)
        model.fit(X_np[train_idx], y_np[train_idx], eval_set=(X_np[val_idx], y_np[val_idx]), verbose=0)
        preds = model.predict(X_np[val_idx])
        scores.append(accuracy_score(y_np[val_idx], preds))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, timeout=1800)
cat_params = study.best_params
cat_params.update({'random_state': 42, 'verbose': 0, 'task_type': 'CPU'})

# Train Final Models
print("🚀 Training base models...")
cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(X_train, y_train)

lgbm_model = LGBMClassifier(random_state=42, n_jobs=4)
lgbm_model.fit(X_train, y_train)

xgb_model = XGBClassifier(random_state=42, n_jobs=4, verbosity=0)
xgb_model.fit(X_train, y_train)

# Stacking Ensemble
print("🔹 Building Stacking Classifier...")
stack = StackingClassifier(
    estimators=[
        ('cat', cat_model),
        ('lgbm', lgbm_model),
        ('xgb', xgb_model)
    ],
    final_estimator=LGBMClassifier(n_estimators=200, learning_rate=0.1),
    n_jobs=1
)
stack.fit(X_train, y_train)

# Evaluation
print("\n📅 Final Evaluation")
preds = stack.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(stack, "models/ckd_stack_model.joblib")
joblib.dump(le, "models/encoder.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(imputer, "models/imputer.joblib")
print("\n📆 All models and encoders saved in 'models/'")
