from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('liver_disease_model.pkl')
le_dict = joblib.load('liver_label_encoders.pkl')

# Define expected feature order (same as during training)
FEATURE_ORDER = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
    'Albumin', 'Albumin_and_Globulin_Ratio'
]

# Clean ambiguous values
def clean_input(data):
    ambiguous = ['?', 'None', 'none', 'Not Mentioned', 'not mentioned', 'N/A', '', 'Unknown', 'unknown', 'No', 'no']
    return {k: (np.nan if str(v).strip() in ambiguous else v) for k, v in data.items()}

# Prepare input
def prepare_input(data):
    data = clean_input(data)
    df = pd.DataFrame([data])

    # Fill missing columns
    for col in FEATURE_ORDER:
        if col not in df:
            df[col] = np.nan

    df = df[FEATURE_ORDER]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill remaining NaNs with median
    df = df.fillna(df.median(numeric_only=True))

    # Encode if needed
    for col, le in le_dict.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str).str.strip())

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        input_df = prepare_input(input_data)
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][prediction]

        return jsonify({
            'prediction': 'Liver Disease' if prediction == 1 else 'No Liver Disease',
            'confidence': round(prob * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Liver Disease Prediction API is running!"

if __name__ == '__main__':
    app.run(debug=True)
