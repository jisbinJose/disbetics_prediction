# Diabetes Prediction System

A machine learning-based web application for predicting diabetes risk using multiple models: Logistic Regression, Polynomial Regression, and K-Nearest Neighbors (KNN).

## Features

- Simple Yes/No Diagnosis using Logistic Regression
- Risk Percentage Assessment using Polynomial Regression
- Similar Cases Comparison using K-Nearest Neighbors
- Modern and responsive web interface
- Real-time predictions

## Project Structure

```
diabetes_prediction/
├── data/
│   └── diabetes_dataset.xlsx
├── models/
│   ├── lr_model.joblib
│   ├── poly_model.joblib
│   ├── knn_model.joblib
│   ├── scaler.joblib
│   └── poly.joblib
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your diabetes dataset in the `data/` folder with the name `diabetes_dataset.xlsx`. The dataset should contain the following columns:
   - Age
   - BMI
   - Blood_Sugar_Level
   - Cholesterol_Level
   - Blood_Pressure
   - Diabetes (target variable: 0 or 1)

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Enter your health information in the form:
   - Age
   - BMI
   - Blood Sugar Level
   - Cholesterol Level
   - Blood Pressure

2. Choose one of the three prediction methods:
   - "Simple Yes/No Diagnosis" for a binary prediction
   - "Show My Risk Percentage" for a risk assessment
   - "Compare with Past Patients" for similar cases analysis

3. View the results displayed below the form

## Technical Details

- The application uses Flask for the web framework
- Models are trained using scikit-learn
- The UI is built with Bootstrap 5
- Models are automatically trained on first run and saved for future use
- Feature scaling is applied to ensure consistent predictions

## Note

This is a demonstration project and should not be used as the sole basis for medical diagnosis. Always consult with healthcare professionals for medical advice. 