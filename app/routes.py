from flask import Blueprint, request, jsonify
from app.model_manager import load_model, predict_yield
from app.preprocess import preprocess_input_data
from app.utils import handle_error
import joblib
import pandas as pd

# Define API Blueprint
api_bp = Blueprint('api', __name__)

# Load pre-trained models
model_p = load_model('models/model_P.pkl')
model_kt = load_model('models/model_KT.pkl')
model_rkt = load_model('models/model_RKT.pkl')

def round_to_nearest_50(value):
    return round(value / 50) * 50

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the API request
        input_data = request.get_json()  # Get JSON data from request

        print(f"Input Data from API:\n{input_data}")
        
        
        new_data = pd.DataFrame(input_data, index=[0])
        
        # Preprocess the input data using the preprocessor
        processed_data = preprocess_input_data(new_data)
        
        print(f"Preprocessed Input Data After Transformation:\n{processed_data}")

        # Predict using the pre-trained models
        prediction_p = predict_yield(model_p, processed_data)
        prediction_kt = predict_yield(model_kt, processed_data)
        prediction_rkt = predict_yield(model_rkt, processed_data)

        # Return the predictions as a JSON response
        return jsonify({
            'පීදුනු කොළ': round_to_nearest_50(prediction_p[0]),
            'කෙටි කොළ': round_to_nearest_50(prediction_kt[0]),
            'රෑන් කෙටි කොළ': round_to_nearest_50(prediction_rkt[0])
        })
        
    except Exception as e:
        return handle_error(e)
