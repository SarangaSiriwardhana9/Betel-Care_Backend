from app.kavindi import DemandPredictionInput, PricePredictionInput, predict_demand_location, predict_market_demand, predict_price
from flask import Blueprint, request, jsonify
from app.model_manager import predict_yield
from app.preprocess import preprocess_input_data
from app.utils import handle_error
import joblib
import pandas as pd
import os

api_bp = Blueprint('api', __name__)

def load_model(model_name):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_name)
    return joblib.load(model_path)

def round_to_nearest_50(value):
    return round(value / 50) * 50

@api_bp.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Welcome to the API!'})


@api_bp.route('/predict/harvest', methods=['POST'])
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
            'P': round_to_nearest_50(prediction_p[0]),
            'KT': round_to_nearest_50(prediction_kt[0]),
            'RKT': round_to_nearest_50(prediction_rkt[0])
        })
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500
    

#! Kavindi Routes starts here ===============>
# Endpoint: Predict price per leaf
@api_bp.post("/market/predict-price")
def predict_price_endpoint():
    data = request.get_json()  # Extract JSON data from the request

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    input_data = PricePredictionInput(**data)  # Convert JSON into an object

    print('Price prediction triggered')

    return jsonify({
        "price": predict_price(
            date=input_data.Date,
            leaf_type=input_data.Leaf_Type,
            leaf_size=input_data.Leaf_Size,
            quality_grade=input_data.Quality_Grade,
            no_of_leaves=input_data.No_of_Leaves,
            location=input_data.Location,
            season=input_data.Season
        )
    })


# Endpoint: Predict highest demand location
@api_bp.post("/market/predict-location")
def predict_location_endpoint():

    data = request.get_json()  # Extract JSON data from the request

    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    input_data = DemandPredictionInput(**data)  # Convert JSON into an object
    
    return jsonify({
        "location": predict_demand_location(
            date=input_data.Date,
            no_of_leaves=input_data.No_of_Leaves,
            leaf_type=input_data.Leaf_Type,
            leaf_size=input_data.Leaf_Size,
            quality_grade=input_data.Quality_Grade
        )
    })

@api_bp.post("/market/predict-market-demand")
def predict_market_demand_endpoint():
    try:
       return predict_market_demand()
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


# Health check endpoint
@api_bp.get("/health")
def health_check():
    return {"status": "API is up and running!"}

# Kavindi Routes ends here ==============================================>
