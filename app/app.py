from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Always load files relative to this app.py file.
# This works even when Flask is launched from another directory.
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Load the previously trained model and preprocessing objects.
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Required model file was not found: {exc.filename}. "
        "Place student_model.pkl, encoder.pkl, and scaler.pkl "
        "in the same folder as app.py."
    ) from exc


# The order must match the model's training data.
FEATURE_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Medu", "Fedu",
    "Mjob", "Fjob", "guardian", "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "subject", "attendance_ratio", "average_grade"
]

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Mjob", "Fjob",
    "guardian", "schoolsup", "famsup", "paid", "activities",
    "higher", "internet", "romantic", "subject"
]

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2"
]

# These two features are calculated by the app, not supplied by the form.
DERIVED_COLS = ["attendance_ratio", "average_grade"]

INPUT_COLUMNS = [
    col for col in FEATURE_COLUMNS if col not in DERIVED_COLS
]


def predict_student(data):
    """Validate inputs, apply saved preprocessing, and predict performance."""

    if not isinstance(data, dict):
        raise ValueError("Please submit student data as a form or JSON object.")

    # Check that every field expected by the model was provided.
    missing = [
        col for col in INPUT_COLUMNS
        if col not in data or data[col] is None or str(data[col]).strip() == ""
    ]

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )

    # Construct the input in the expected order.
    input_df = pd.DataFrame([
        {col: data[col] for col in INPUT_COLUMNS}
    ])

    # Convert numeric form values from strings to numbers.
    for col in NUMERIC_COLS:
        input_df[col] = pd.to_numeric(input_df[col], errors="raise")

    # Calculate derived features exactly as specified by this pipeline.
    total_days = 90
    input_df["attendance_ratio"] = (
        1 - input_df["absences"] / total_days
    )
    input_df["average_grade"] = (
        input_df["G1"] + input_df["G2"]
    ) / 2

    # Use the FITTED encoder; do not fit it on the student's row.
    input_df[CATEGORICAL_COLS] = encoder.transform(
        input_df[CATEGORICAL_COLS]
    )

    # Keep the same feature order used during training.
    input_df = input_df[FEATURE_COLUMNS]

    # Use the FITTED scaler; do not fit it on the student's row.
    scaled_features = scaler.transform(input_df)

    # Generate the prediction.
    prediction = model.predict(scaled_features)[0]

    # Convert NumPy scalar values to standard Python values when needed.
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return prediction


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle predictions submitted through the HTML form."""
    try:
        data = request.form.to_dict()
        result = predict_student(data)

        return render_template(
            "index.html",
            prediction_text=f"Student Performance: {result}"
        )

    except Exception as exc:
        app.logger.exception("Form prediction failed")
        return render_template(
            "index.html",
            prediction_text=f"Prediction error: {str(exc)}"
        ), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Handle predictions submitted as JSON."""
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Send a valid JSON object containing student data."
            }), 400

        result = predict_student(data)

        return jsonify({"prediction": result})

    except Exception as exc:
        app.logger.exception("API prediction failed")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    # Debug mode is suitable only for local development.
    app.run(debug=True)
