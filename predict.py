#prediction code to check working of pipeline

import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

model = joblib.load("models/ckd_stack_model.joblib")
scaler = joblib.load("models/scaler.joblib")
imputer = joblib.load("models/imputer.joblib")
selected_features = joblib.load("models/selected_features.joblib")
encoder = joblib.load("models/encoder.joblib")
all_feature_names = list(imputer.feature_names_in_)

# Your test row as a DataFrame (with string values for categoricals)
data = [[65,145,1.015,2,0,'normal','abnormal','not present','not present',140,55,2.2,138,4.8,11.5,35,7500,4.2,
         'yes','yes','no','poor','yes','yes',45,0.8,1200,3.8,220,85,8.8,4.2,'yes','no',28,2,10,12,1.8,'normal',4.5,3.2,
         11.5*2.2, 55/(2.2+1e-5), 45/(65+1e-5), 0]]

df = pd.DataFrame(data, columns=all_feature_names)

# Encode categorical columns using your saved encoder
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Target']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))
if df['Target'].dtype == 'object':
    df['Target'] = encoder.fit_transform(df['Target'].astype(str))

# Reorder columns to match imputer's fit order
df = df[all_feature_names]

# Impute
df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)

# Drop 'Target' before scaling
df_for_scaler = df_imputed.drop(columns=['Target'])

# Scale
df_scaled = pd.DataFrame(scaler.transform(df_for_scaler), columns=df_for_scaler.columns)

# Remove 'Target' from selected_features if present
features_for_model = [f for f in selected_features if f != 'Target']

# Select only the features used by the model
X_selected = df_scaled.loc[:, features_for_model]

print("X_selected columns:", list(X_selected.columns))
print("features_for_model:", features_for_model)
print(model.predict(X_selected))