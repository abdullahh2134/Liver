# app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load trained model and label encoders
model = joblib.load('liver_disease_model.pkl')
label_encoders = joblib.load('liver_label_encoders.pkl')

EXPECTED_COLUMNS = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase', 'Total_Protiens',
    'Albumin', 'Albumin_and_Globulin_Ratio'
]

@app.route('/')
def home():
    return "✅ Liver Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        df = df[EXPECTED_COLUMNS]
        prediction = model.predict(df)[0]
        result = 'Liver Disease Detected' if prediction == 1 else 'No Liver Disease Detected'

        return jsonify({
            'prediction': int(prediction),
            'result': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
