from flask import Blueprint, request, jsonify
from app.model_manager import predict_yield
from app.preprocess import preprocess_input_data
from app.utils import handle_error
import joblib
import pandas as pd
import os

# Import your additional blueprints
from app.watering_routes import watering_bp
from app.fertilizing_routes import fertilizing_bp
from app.protection_routes import protection_bp

api_bp = Blueprint('api', __name__)

# Register your additional blueprints with the main api_bp
api_bp.register_blueprint(watering_bp, url_prefix='/watering')
api_bp.register_blueprint(fertilizing_bp, url_prefix='/fertilizing')
api_bp.register_blueprint(protection_bp, url_prefix='/protection')

def load_model(model_name):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
    return joblib.load(model_path)

def round_to_nearest_50(value):
    return round(value / 50) * 50

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'Invalid or missing JSON input'}), 400

        print(f"Received Input Data: {input_data}")

        new_data = pd.DataFrame(input_data, index=[0])
        processed_data = preprocess_input_data(new_data)
        print(f"Processed Data: {processed_data}")

        # ✅ Load models on demand instead of at startup
        model_p = load_model('model_P.pkl')
        model_kt = load_model('model_KT.pkl')
        model_rkt = load_model('model_RKT.pkl')

        prediction_p = predict_yield(model_p, processed_data)
        prediction_kt = predict_yield(model_kt, processed_data)
        prediction_rkt = predict_yield(model_rkt, processed_data)

        return jsonify({
            'පීදුනු කොළ': round_to_nearest_50(prediction_p[0]),
            'කෙටි කොළ': round_to_nearest_50(prediction_kt[0]),
            'රෑන් කෙටි කොළ': round_to_nearest_50(prediction_rkt[0])
        })
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500