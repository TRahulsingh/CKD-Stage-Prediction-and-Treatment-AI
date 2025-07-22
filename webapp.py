import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
from fpdf import FPDF

# --- REMOVES WARNINGS ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message='.*use_column_width.*')
# st.set_option('deprecation.showPyplotGlobalUse', False)

# --- PAGE CONFIG ---
st.set_page_config(page_title="CKD Stage Prediction and Treatment AI", layout="wide")
st.title("⚕️ CKD Stage Prediction and Treatment AI")
st.markdown("A comprehensive tool to determine the clinical stage of Chronic Kidney Disease based on patient data, primarily using the eGFR value as per medical guidelines.")

# --- LOAD MODELS & ASSETS ---
@st.cache_resource
def load_assets():
    """Loads all required models and preprocessing objects."""
    model = joblib.load("models/ckd_stack_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    target_encoder = joblib.load("models/encoder.joblib")
    imputer = joblib.load("models/imputer.joblib")
    selected_features = joblib.load("models/selected_features.joblib")
    all_feature_names = list(imputer.feature_names_in_)
    return model, scaler, target_encoder, imputer, selected_features, all_feature_names

model, scaler, target_encoder, imputer, selected_features, all_feature_names = load_assets()

if 'Target' in selected_features:
    selected_features = [f for f in selected_features if f != 'Target']

# --- [EXPANDED] CLINICAL STAGE DEFINITIONS & GUIDANCE ---
ckd_stage_guidance = {
    'Stage 1': {
        'description': "Kidney damage with normal or high function (eGFR ≥ 90).",
        'goals': ["Slow disease progression", "Reduce cardiovascular disease risk"],
        'dietary_focus': ["Maintain a balanced, heart-healthy diet", "Avoid high-sodium processed foods", "Ensure adequate hydration"],
        'lifestyle': ["Engage in regular exercise (at least 150 mins/week)", "Quit smoking completely", "Avoid NSAID painkillers (e.g., Ibuprofen, Naproxen) unless approved by a doctor"],
        'monitoring': ["Annual check-up with blood pressure monitoring", "Urine test for albumin", "Blood test for creatinine to calculate eGFR"],
        'complications': ["Generally none, but the underlying cause (e.g., diabetes, hypertension) needs aggressive management."]
    },
    'Stage 2': {
        'description': "Mild loss of kidney function (eGFR 60-89).",
        'goals': ["Continue to slow progression", "Aggressively manage blood pressure and blood sugar"],
        'dietary_focus': ["Implement a low-sodium diet (< 2,300 mg/day)", "Continue heart-healthy eating patterns"],
        'lifestyle': ["All Stage 1 recommendations apply and are critical."],
        'monitoring': ["Follow-up every 6-12 months", "Regular monitoring of blood pressure, eGFR, and urine albumin levels"],
        'complications': ["Increased risk of hypertension. Early signs of bone metabolism changes may appear."]
    },
    'Stage 3a': {
        'description': "Mild to moderate loss of kidney function (eGFR 45-59).",
        'goals': ["Prevent and treat complications", "Prepare for more intensive management"],
        'dietary_focus': ["Consult a renal dietitian", "Limit sodium (< 2,000 mg/day)", "May need to limit phosphorus and protein based on lab results"],
        'lifestyle': ["Moderate, consistent exercise is key", "Close self-monitoring of blood pressure and blood sugar (if diabetic)"],
        'monitoring': ["Referral to a Nephrologist (kidney specialist) is standard", "Check-ups every 3-6 months", "Blood tests for phosphorus, calcium, PTH, and hemoglobin"],
        'complications': ["Anemia (low red blood cell count)", "Early bone disease", "Hypertension becomes more common and harder to control"]
    },
    'Stage 3b': {
        'description': "Moderate to severe loss of kidney function (eGFR 30-44).",
        'goals': ["Intensively manage complications", "Educate on renal replacement therapies"],
        'dietary_focus': ["Stricter adherence to renal diet (protein, sodium, potassium, phosphorus limits)", "Fluid intake may need monitoring"],
        'lifestyle': ["All previous recommendations are mandatory.", "Avoid exhaustion; balance activity and rest."],
        'monitoring': ["Nephrologist follow-up every 3 months", "Frequent lab work to manage complications"],
        'complications': ["Anemia and bone disease are more pronounced", "Increased risk of acidosis (build-up of acid in the blood)"]
    },
    'Stage 4': {
        'description': "Severe loss of kidney function (eGFR 15-29).",
        'goals': ["Prepare for kidney failure treatment (dialysis or transplant)"],
        'dietary_focus': ["Very strict diet is essential.", "Work closely with a renal dietitian."],
        'lifestyle': ["Fluid restriction is common.", "Preserve remaining kidney function by following medical advice perfectly."],
        'monitoring': ["Monthly or bi-monthly visits to the nephrologist", "Evaluation for dialysis access placement (e.g., fistula) or transplant listing"],
        'complications': ["High risk of all CKD complications", "Symptoms like fatigue, swelling, and nausea become more common"]
    },
    'Stage 5': {
        'description': "Kidney failure / End-Stage Renal Disease (eGFR < 15).",
        'goals': ["Sustain life with renal replacement therapy"],
        'dietary_focus': ["Follow a specific dialysis-friendly diet, which may differ depending on the type of dialysis", "Strict fluid, sodium, potassium, and phosphorus limits"],
        'lifestyle': ["Adherence to dialysis schedule is life-sustaining.", "Manage symptoms and maintain quality of life."],
        'monitoring': ["Constant medical supervision is required as part of dialysis or post-transplant care."],
        'complications': ["This is a life-threatening condition requiring constant medical care to manage fluid overload, electrolyte imbalances, and waste product buildup."]
    }
}


# --- HELPER FUNCTIONS ---
def get_stage_from_egfr(egfr_value):
    """Returns the clinical CKD stage based on eGFR value."""
    if egfr_value >= 90:
        return 'Stage 1'
    elif 60 <= egfr_value < 90:
        return 'Stage 2'
    elif 45 <= egfr_value < 60:
        return 'Stage 3a'
    elif 30 <= egfr_value < 45:
        return 'Stage 3b'
    elif 15 <= egfr_value < 30:
        return 'Stage 4'
    else: # eGFR < 15
        return 'Stage 5'

def preprocess(df):
    """Applies the full, correct preprocessing pipeline to new data."""
    mapping = {
        'yes': 1, 'no': 0, 'normal': 1, 'abnormal': 0,
        'present': 1, 'not present': 0, 'good': 1, 'poor': 0
    }
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Target':
             df[col] = df[col].str.lower().str.strip().map(mapping)
    if 'Serum creatinine (mg/dl)' in df.columns and 'Hemoglobin level (gms)' in df.columns:
        df['Creatinine_Hemoglobin'] = df['Serum creatinine (mg/dl)'] * df['Hemoglobin level (gms)']
    if 'Blood urea (mg/dl)' in df.columns and 'Serum creatinine (mg/dl)' in df.columns:
        df['BUN_Creatinine_Ratio'] = df['Blood urea (mg/dl)'] / (df['Serum creatinine (mg/dl)'] + 1e-5)
    if 'Estimated Glomerular Filtration Rate (eGFR)' in df.columns and 'Age of the patient' in df.columns:
        df['eGFR_by_Age'] = df['Estimated Glomerular Filtration Rate (eGFR)'] / (df['Age of the patient'] + 1e-5)
    if 'Target' in df.columns:
        if df['Target'].dtype == 'object':
            df['Target'] = target_encoder.transform(df['Target'])
    else:
        df['Target'] = 0
    for col in all_feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[all_feature_names]
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    df_for_scaler = df_imputed.drop(columns=['Target'])
    df_scaled = pd.DataFrame(scaler.transform(df_for_scaler), columns=df_for_scaler.columns)
    X_selected = df_scaled[selected_features]
    return X_selected

def generate_pdf(patient_data, stage):
    """Generates a downloadable PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Clinical CKD Stage Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Report Date: {datetime.datetime.now().strftime('%d-%m-%Y')}", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Determined Clinical Stage: {stage}", ln=True)
    guidance = ckd_stage_guidance.get(stage, {})
    if 'description' in guidance:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 8, f"({guidance['description']})", ln=True)
    pdf.ln(5)

    def write_section(title, content_list):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", '', 11)
        for item in content_list:
            pdf.multi_cell(0, 6, f"  - {item}")
        pdf.ln(3)

    if guidance:
        write_section("Key Goals:", guidance.get('goals', []))
        write_section("Dietary Focus:", guidance.get('dietary_focus', []))
        write_section("Lifestyle Recommendations:", guidance.get('lifestyle', []))
        write_section("Key Complications to Monitor:", guidance.get('complications', []))

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, "Disclaimer:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, "This report is for informational purposes only. The clinical stage is determined by the eGFR value as per standard medical guidelines. It is not a substitute for professional medical advice. Always consult a qualified nephrologist for diagnosis, treatment, and management plans.")
    
    pdf_path = f"ckd_report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(pdf_path)
    return pdf_path

# --- UI LAYOUT ---

# --- SECTION 1: CSV UPLOAD ---
st.header("📄 Predict from CSV File")
csv_file = st.file_uploader("Upload a .csv file with patient data", type="csv")

if csv_file:
    try:
        df_csv = pd.read_csv(csv_file)
        st.dataframe(df_csv)
        raw_egfr_values = df_csv['Estimated Glomerular Filtration Rate (eGFR)'].tolist()
        st.success("✅ Staging Complete!")
        
        final_stages = [get_stage_from_egfr(egfr) for egfr in raw_egfr_values]
        
        df_csv['Determined_Stage'] = final_stages
        csv_output = df_csv.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Staged CSV", data=csv_output, file_name="staged_predictions.csv", mime="text/csv")

        for i, stage in enumerate(final_stages):
            st.write(f"**Patient {i+1} → Determined Clinical Stage: {stage}**")
            guidance = ckd_stage_guidance.get(stage)
            if guidance:
                with st.expander(f"🔬 View Guidance for Patient {i+1} ({stage})"):
                    st.markdown(f"**Description:** {guidance['description']}")
                    
                    st.markdown("**Key Goals:**")
                    for item in guidance['goals']:
                        st.markdown(f"- {item}")

                    st.markdown("**Dietary Focus:**")
                    for item in guidance['dietary_focus']:
                        st.markdown(f"- {item}")

                    st.markdown("**Lifestyle Recommendations:**")
                    for item in guidance['lifestyle']:
                        st.markdown(f"- {item}")
                        
                    st.markdown("**Key Complications to Monitor:**")
                    for item in guidance['complications']:
                        st.markdown(f"- {item}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")

# --- SECTION 2: MANUAL DATA ENTRY (IN AN EXPANDER) ---
with st.expander("✍️ Or Enter Patient Data Manually"):
    if 'saved_inputs' not in st.session_state:
        st.session_state.saved_inputs = []

    with st.form("manual_input_form"):
        manual_input = {}
        st.write("Fill in the patient's lab values and information below.")
        cols = st.columns(3)
        feature_list = [f for f in all_feature_names if f not in ["Creatinine_Hemoglobin", "BUN_Creatinine_Ratio", "eGFR_by_Age", "Target"]]
        for i, feature in enumerate(feature_list):
            with cols[i % 3]:
                if feature in ["Hypertension (yes/no)", "Diabetes mellitus (yes/no)", "Coronary artery disease (yes/no)", "Pedal edema (yes/no)", "Anemia (yes/no)", "Family history of chronic kidney disease", "Smoking status"]:
                    manual_input[feature] = st.selectbox(f"{feature}", ["no", "yes"], key=feature)
                elif feature in ["Pus cell clumps in urine", "Bacteria in urine"]:
                    manual_input[feature] = st.selectbox(f"{feature}", ["not present", "present"], key=feature)
                elif feature == "Appetite (good/poor)":
                    manual_input[feature] = st.selectbox(f"{feature}", ["good", "poor"], key=feature)
                elif feature in ["Red blood cells in urine", "Pus cells in urine", "Urinary sediment microscopy results"]:
                    manual_input[feature] = st.selectbox(f"{feature}", ["normal", "abnormal"], key=feature)
                else:
                    manual_input[feature] = st.number_input(f"{feature}", value=45.0 if 'eGFR' in feature else 0.0, format="%.2f", key=feature)
        submitted = st.form_submit_button("🩺 Determine Clinical Stage")

    if submitted:
        try:
            raw_egfr = manual_input['Estimated Glomerular Filtration Rate (eGFR)']
            final_stage = get_stage_from_egfr(raw_egfr)
            st.success(f"✅ Determined Clinical Stage: **{final_stage}**")
            
            guidance = ckd_stage_guidance.get(final_stage)
            if guidance:
                st.markdown("---")
                st.markdown(f"### 🔬 Detailed Guidance for {final_stage}")
                st.markdown(f"**Description:** {guidance['description']}")

                st.markdown("**Key Goals:**")
                for item in guidance['goals']:
                    st.markdown(f"- {item}")

                st.markdown("**Dietary Focus:**")
                for item in guidance['dietary_focus']:
                    st.markdown(f"- {item}")

                st.markdown("**Lifestyle Recommendations:**")
                for item in guidance['lifestyle']:
                    st.markdown(f"- {item}")
                    
                st.markdown("**Key Complications to Monitor:**")
                for item in guidance['complications']:
                    st.markdown(f"- {item}")
            
            save_input = manual_input.copy()
            save_input['Determined_Stage'] = final_stage
            save_input['Timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.session_state.saved_inputs.insert(0, save_input)
            
            pdf_path = generate_pdf(save_input, final_stage)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Full Report (PDF)", f, file_name=f"CKD_Report_{final_stage}.pdf")
            os.remove(pdf_path)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")

# --- SECTION 3: MODEL EXPLAINABILITY ---
st.header("📊 Model Explainability")
st.markdown("These plots show which features were most influential for each of the base models in the prediction stack. This provides insight into the model's internal risk assessment which runs in the background.")

col1, col2, col3 = st.columns(3)
def show_shap_plot(model_name, column):
    path = f"shap_plots/shap_summary_{model_name}.png"
    if os.path.exists(path):
        column.image(path, caption=f"{model_name.upper()} SHAP Plot", use_container_width=True)
        with open(path, "rb") as f:
            column.download_button(f"Download {model_name.upper()} SHAP Plot", f, file_name=f"{model_name}_shap.png")
    else:
        column.warning(f"{model_name.upper()} SHAP plot not found. Ensure it exists in the 'shap_plots/' directory.")
        
show_shap_plot("cat", col1)
show_shap_plot("lgbm", col2)
show_shap_plot("xgb", col3)

st.markdown("---")

# --- SECTION 4: PREDICTION HISTORY & FOOTER ---
if st.session_state.saved_inputs:
    st.header("📜 Recent Predictions History")
    with st.expander("Click to view past manual predictions"):
        df_saved = pd.DataFrame(st.session_state.saved_inputs)
        st.dataframe(df_saved)
        if st.button("Clear Prediction History"):
            st.session_state.saved_inputs = []
            st.rerun()

st.markdown("©️ **Disclaimer**: This tool is for educational purposes and not a substitute for professional medical advice. The clinical stage is determined from the eGFR value as per standard medical guidelines. Always consult a qualified medical professional for diagnosis and treatment.")
st.markdown("🔗 [National Kidney Foundation](https://www.kidney.org)")
