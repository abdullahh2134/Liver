from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Load the trained model and label encoders
model = joblib.load('liver_disease_model.pkl')
label_encoders = joblib.load('liver_label_encoders.pkl')

# Define the Flask app
app = Flask(__name__)

# Expected feature columns in correct order (must match training)
EXPECTED_COLUMNS = [
   'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
      'Alkaline_Phosphatase', 'Alanine_Aminotransferase',
      'Aspartate_Aminotransferase', 'Total_Protiens',
      'Albumin'
]

@app.route('/')
def home():
    return "✅ Liver Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()

        # Convert input into DataFrame
        df = pd.DataFrame([data])

        # Apply label encoding to categorical fields
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # Ensure correct column order
        df = df[EXPECTED_COLUMNS]

        # Make prediction
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
