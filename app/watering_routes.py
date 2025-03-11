from flask import Blueprint, request, jsonify
import joblib
import os
import json
import numpy as np
from datetime import datetime
from app.watering_model import load_watering_model, prepare_features, get_watering_recommendation

watering_bp = Blueprint('watering', __name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@watering_bp.route('/predict', methods=['POST'])
def predict_watering():
    try:
        # Get input data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON input'}), 400
        
        # Extract parameters
        location = data.get('location', 'PUTTALAM')  # Default to PUTTALAM if not specified
        rainfall = float(data.get('rainfall', 0.0))
        min_temp = float(data.get('min_temp', 25.0))
        max_temp = float(data.get('max_temp', 30.0))
        crop_stage = int(data.get('crop_stage', 5))  # Default to mature plants
        month = data.get('month', None)
        
        # Validate location
        if location not in ['PUTTALAM', 'KURUNEGALA']:
            return jsonify({'error': 'Location must be either PUTTALAM or KURUNEGALA'}), 400
        
        # If month not provided, use current month
        if month is None:
            month = datetime.now().month
        else:
            month = int(month)
        
        # Load the watering model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'betel_watering_model.pkl')
        model = joblib.load(model_path)
        
        # Prepare features
        features = prepare_features(location, rainfall, min_temp, max_temp, crop_stage, month)
        
        # Get recommendation
        recommendation = get_watering_recommendation(model, features)
        
        # Add location to response
        response = {
            'location': location,
            'watering_recommendation': recommendation['recommendation'],
            'water_amount': recommendation['water_amount'],
            'confidence': recommendation['confidence'],
            'consecutive_dry_days': recommendation['consecutive_dry_days'],
            'probabilities': recommendation['probabilities']
        }
        
        # Use the custom JSON encoder to handle any NumPy types
        return json.loads(json.dumps(response, cls=NumpyEncoder))
        
    except Exception as e:
        print("Error in watering prediction:", str(e))
        return jsonify({'error': str(e)}), 500