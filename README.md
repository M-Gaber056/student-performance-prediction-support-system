# Student Performance Prediction Support System

A machine learning project that explores student academic performance, identifies performance-risk categories, and provides a web interface for generating predictions based on student information.

## Project Overview

The Student Performance Prediction Support System was developed as a final project for the MAIM program. It combines data preprocessing, exploratory data analysis, machine learning, and text sentiment analysis to investigate factors associated with student academic performance.

The project also includes a Flask-based web application intended to allow users to submit student information and receive a model prediction.

## Objectives

- Analyze student academic and demographic information.
- Explore relationships between student characteristics, study habits, and academic outcomes.
- Engineer features to support machine learning analysis.
- Classify students into performance-risk categories based on final grades.
- Explore sentiment analysis on student feedback.
- Develop a web interface for interacting with the prediction model.

## Technologies Used

- **Python** — core programming language
- **Pandas and NumPy** — data manipulation and preprocessing
- **Matplotlib and Seaborn** — data visualization
- **Scikit-learn** — preprocessing, machine learning, and model evaluation
- **TensorFlow / Keras** — neural network experimentation
- **NLTK and TextBlob** — text processing and sentiment analysis
- **Hugging Face Transformers** — NLP-related library exploration
- **Flask** — web application
- **Joblib** — saving and loading model artifacts
- **Jupyter Notebook** — experimentation and analysis

## Repository Structure

```text
student-performance-prediction-support-system/
│
├── MAIM_FinalProject.ipynb
├── Report.docx
├── Final_Presentation.pptx
├── README.md
├── requirements.txt
├── student_model.pkl
│
├── raw-data/
│   ├── student-mat.csv
│   └── student-por.csv
│
└── app/
    ├── app.py
    ├── student_model.pkl
    ├── encoder.pkl
    ├── scaler.pkl
    └── templates/
        └── index.html
```

## Dataset

The project uses two CSV datasets containing student information and academic grades:

- `student-mat.csv` — student mathematics performance
- `student-por.csv` — student Portuguese language performance

The datasets include demographic information, family background, study habits, school-related support, lifestyle indicators, absences, and grades.

The datasets originate from the UCI Machine Learning Repository's Student Performance dataset.

Dataset reference: [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)

## Methodology

### 1. Data Preprocessing

The notebook loads the datasets, corrects column names, and prepares the data for analysis and modeling.

### 2. Exploratory Data Analysis

Statistical analysis and visualizations are used to investigate student characteristics, academic grades, and potential relationships between features and performance.

### 3. Feature Engineering

Additional features are created from existing information, including an attendance ratio and an average of the first- and second-period grades.

### 4. Performance-Risk Classification

The project defines performance categories using the final grade (G3):

- **High Risk:** final grade of 0–10
- **Medium Risk:** final grade of 11–14
- **Low Risk:** final grade of 15–20

The notebook explores classification approaches and evaluates model performance using appropriate classification metrics.

The first- and second-period grades (G1 and G2) are available as earlier academic indicators. The final grade (G3), which defines the target categories, should not be included in the model's input features.

### 5. Sentiment Analysis

The notebook also explores sentiment analysis of student feedback. The interpretation of these results depends on how the feedback data was generated; synthetic feedback should not be treated as independent evidence of real student sentiment.

### 6. Web Application

A Flask application provides a web interface for entering student information and requesting a prediction from a saved model.

The application and its preprocessing pipeline must use the same feature definitions, categorical encodings, and scaling transformations used during model training.

## Installation

### Prerequisites

- Python 3.10 or 3.11 is recommended.
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/M-Gaber056/student-performance-prediction-support-system.git
cd student-performance-prediction-support-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Notebook

Open Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

Open `MAIM_FinalProject.ipynb` and execute the cells in order.

Keep the `raw-data` directory in its original location so that the notebook's relative dataset paths resolve correctly.

Some NLP workflows may require downloading the relevant NLTK resources before execution.

## Running the Web Application

The Flask application is located in the `app` directory.

```bash
cd app
python app.py
```

Open the local address displayed in the terminal in your web browser.

**Important:** The saved model and preprocessing objects must match the training pipeline. Verify that the model loads correctly, that the form supplies every required feature, and that predictions work with the fitted encoder and scaler before relying on the application.

## Project Documentation

Additional project materials are included in this repository:

- `Report.docx` — written project report
- `Final_Presentation.pptx` — final project presentation
- `MAIM_FinalProject.ipynb` — notebook containing the analysis and experiments

## Limitations

- Predictions are based on the available dataset and are not definitive assessments of an individual student's ability.
- Performance metrics depend on the preprocessing, train/test split, feature selection, and evaluation methodology.
- The model should be evaluated again after any change to the target definition or input features.
- Synthetic feedback cannot establish real-world relationships between sentiment and academic performance.
- The web application's predictions should be verified after aligning its preprocessing steps with those used during model training.

## Intended Use

This project was developed for educational and research purposes to demonstrate data analysis, machine learning, and web application development. It should not be used as the sole basis for decisions about students.

