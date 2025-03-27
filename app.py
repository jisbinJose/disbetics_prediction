from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load and prepare data
def load_data():
    try:
        data = pd.read_excel('data/diabetes_dataset.xlsx')
        X = data[['Age', 'BMI', 'Blood Sugar Level', 'Cholesterol Level', 'Blood Pressure']]
        y = data['Diabetes']
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Train models
def train_models():
    try:
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_accuracy = lr_model.score(X_test_scaled, y_test) * 100
        
        # Train Polynomial Regression
        poly = PolynomialFeatures(degree=2)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        poly_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        poly_model.fit(X_train_poly, y_train)
        poly_accuracy = poly_model.score(X_test_poly, y_test) * 100
        
        # Train KNN
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train_scaled, y_train)
        knn_accuracy = knn_model.score(X_test_scaled, y_test) * 100
        
        logger.info(f"Model Accuracies:")
        logger.info(f"Logistic Regression: {lr_accuracy:.2f}%")
        logger.info(f"Polynomial Regression: {poly_accuracy:.2f}%")
        logger.info(f"KNN: {knn_accuracy:.2f}%")
        
        return lr_model, poly_model, knn_model, scaler, poly, lr_accuracy, poly_accuracy, knn_accuracy
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load or train models
try:
    logger.info("Loading existing models...")
    lr_model = joblib.load('models/lr_model.joblib')
    poly_model = joblib.load('models/poly_model.joblib')
    knn_model = joblib.load('models/knn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    poly = joblib.load('models/poly.joblib')
    lr_accuracy = joblib.load('models/lr_accuracy.joblib')
    poly_accuracy = joblib.load('models/poly_accuracy.joblib')
    knn_accuracy = joblib.load('models/knn_accuracy.joblib')
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.info("Training new models...")
    try:
        lr_model, poly_model, knn_model, scaler, poly, lr_accuracy, poly_accuracy, knn_accuracy = train_models()
        joblib.dump(lr_model, 'models/lr_model.joblib')
        joblib.dump(poly_model, 'models/poly_model.joblib')
        joblib.dump(knn_model, 'models/knn_model.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        joblib.dump(poly, 'models/poly.joblib')
        joblib.dump(lr_accuracy, 'models/lr_accuracy.joblib')
        joblib.dump(poly_accuracy, 'models/poly_accuracy.joblib')
        joblib.dump(knn_accuracy, 'models/knn_accuracy.joblib')
        logger.info("Models trained and saved successfully!")
    except Exception as e:
        logger.error(f"Error in model initialization: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input/<method>')
def input_form(method):
    if method not in ['logistic', 'polynomial', 'knn']:
        return 'Invalid method', 400
    return render_template('input_form.html', method=method)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        method = data.get('method')
        
        if not method:
            return jsonify({'error': 'Method not specified'}), 400
        
        # Initialize features array with default values
        features = np.array([[0, 0, 0, 0, 0]])
        
        # Update features based on the method
        if method == 'logistic':
            features[0][1] = float(data['bmi'])  # BMI
            features[0][2] = float(data['blood_sugar'])  # Blood Sugar
        elif method == 'polynomial':
            features[0][2] = float(data['blood_sugar'])  # Only Blood Sugar
        else:  # knn
            features[0][0] = float(data['age'])  # Age
            features[0][1] = float(data['bmi'])  # BMI
            features[0][2] = float(data['blood_sugar'])  # Blood Sugar
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get predictions based on the method
        if method == 'logistic':
            lr_pred = lr_model.predict(features_scaled)[0]
            return jsonify({
                'logistic_prediction': int(lr_pred),
                'model_accuracy': round(lr_accuracy, 2)
            })
        elif method == 'polynomial':
            # Transform features to polynomial features
            features_poly = poly.transform(features_scaled)
            # Get probability of positive class (diabetes)
            poly_prob = poly_model.predict_proba(features_poly)[0][1]
            # Convert to percentage and round to 2 decimal places
            risk_percentage = round(poly_prob * 100, 2)
            
            # Adjust risk based on blood sugar level
            blood_sugar = float(data['blood_sugar'])
            if blood_sugar > 140:
                risk_percentage = max(risk_percentage, 30)  # Minimum 30% risk for high blood sugar
            
            return jsonify({
                'risk_percentage': risk_percentage,
                'model_accuracy': round(poly_accuracy, 2)
            })
        else:  # knn
            distances, indices = knn_model.kneighbors(features_scaled, n_neighbors=5)
            similar_cases = distances[0].tolist()
            
            # Calculate average distance to determine risk level
            avg_distance = sum(similar_cases) / len(similar_cases)
            if avg_distance < 0.5:
                risk_level = "High chance of diabetes"
            elif avg_distance < 1.0:
                risk_level = "Moderate chance of diabetes"
            else:
                risk_level = "Low chance of diabetes"
            
            return jsonify({
                'risk_level': risk_level,
                'similarity_score': round(1 - avg_distance, 2),
                'model_accuracy': round(knn_accuracy, 2)
            })
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 