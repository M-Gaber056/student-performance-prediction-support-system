from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

app = Flask(__name__)

# Load your trained model
model = joblib.load('student_model.pkl')

# Define the column structure (must match your training data)
feature_columns = [
    'school', 'sex', 'age', 'address', 'famsize', 'Medu', 'Fedu', 
    'Mjob', 'Fjob', 'guardian', 'traveltime', 'studytime', 'failures',
    'schoolsup', 'famsup', 'paid', 'activities', 'higher', 'internet',
    'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2', 'subject', 'attendance_ratio', 'average_grade'
]

# Categorical columns that need encoding (with the same mapping as during training)
categorical_cols = ['school', 'sex', 'address', 'famsize', 'Mjob', 'Fjob', 
                   'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                   'higher', 'internet', 'romantic', 'subject']

# Initialize encoders and scaler (you might need to save and load these too)
# For a complete solution, you should save and load your fitted encoder and scaler
encoder = OrdinalEncoder()
scaler = MinMaxScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Create a DataFrame with the expected structure
        input_df = pd.DataFrame([data], columns=feature_columns)
        
        # Convert numeric columns
        numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                       'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                       'absences', 'G1', 'G2']
        
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col])
        
        # Calculate derived features (must match your training pipeline)
        total_days = 90
        input_df['attendance_ratio'] = 1 - (input_df['absences'] / total_days)
        input_df['average_grade'] = (input_df['G1'] + input_df['G2']) / 2
        
        # Encode categorical variables (Note: This should use the same encoder as training)
        # In a real scenario, you should save and load the fitted encoder
        input_df[categorical_cols] = encoder.fit_transform(input_df[categorical_cols])
        
        # Scale the features (Note: This should use the same scaler as training)
        # In a real scenario, you should save and load the fitted scaler
        scaled_features = scaler.fit_transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Return result
        result = prediction[0]
        return render_template('index.html', prediction_text=f'Student Performance: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        # Similar processing as above
        # ...
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)