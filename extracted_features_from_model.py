# extract_features_from_model.py

import joblib

# Load the trained model
model = joblib.load("models/ckd_stack_model.joblib")

# Get features from the training data used in the final model
cat_model = model.named_estimators_['cat']
feature_names = list(cat_model.feature_names_)

# If 'Target' is not in the list, add it at the end
if 'Target' not in feature_names:
    feature_names.append('Target')

# Save the selected features as a list
joblib.dump(feature_names, "models/selected_features.joblib")
print(f"✅ Extracted {len(feature_names)} features used during training (including 'Target').")