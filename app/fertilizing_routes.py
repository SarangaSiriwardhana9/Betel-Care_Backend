from flask import Blueprint, request, jsonify
import json
import numpy as np
from datetime import datetime
from app.fertilizing_model import predict_7day_fertilizing_suitability, predict_today_fertilizing_suitability, convert_numpy_types
from app.fertilizer_planner import fertilizer_planning_api

fertilizing_bp = Blueprint('fertilizing', __name__)

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

@fertilizing_bp.route('/predict', methods=['POST'])
def predict_fertilizing():
    try:
        # Get input data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON input'}), 400
        
        # Extract parameters with default values
        location = data.get('location', 'PUTTALAM')  # Default to PUTTALAM if not specified
        rainfall_forecast = data.get('rainfall_forecast', [0.0] * 7)  # Default to 7 dry days
        
        # Validate location
        if location not in ['PUTTALAM', 'KURUNEGALA']:
            return jsonify({'error': 'Location must be either PUTTALAM or KURUNEGALA'}), 400
        
        # Validate rainfall forecast
        if not isinstance(rainfall_forecast, list):
            return jsonify({'error': 'rainfall_forecast must be a list of 7 values'}), 400
        
        # Ensure we have 7 days of forecast
        if len(rainfall_forecast) < 7:
            # If less than 7 days provided, pad with zeros
            rainfall_forecast = rainfall_forecast + [0.0] * (7 - len(rainfall_forecast))
        elif len(rainfall_forecast) > 7:
            # If more than 7 days provided, truncate
            rainfall_forecast = rainfall_forecast[:7]
        
        # Convert to float (in case they're strings in the JSON)
        rainfall_forecast = [float(r) for r in rainfall_forecast]
        
        # Get fertilizing recommendations
        recommendations = predict_7day_fertilizing_suitability(location, rainfall_forecast)
        
        # Add location and current time information to response
        current_time = datetime.now()
        is_after_six_pm = current_time.hour >= 18
        
        response = {
            'location': location,
            'forecast_start_date': current_time.strftime('%Y-%m-%d'),
            'current_time': current_time.strftime('%H:%M:%S'),
            'is_after_six_pm': is_after_six_pm,
            'daily_recommendations': recommendations
        }
        
        # Find best day for fertilizing
        suitable_days = [day for day in recommendations if day.get('suitable_for_fertilizing', False)]
        if suitable_days:
            best_day = max(suitable_days, key=lambda x: x.get('confidence', 0))
            response['best_day'] = best_day.get('date')
            response['best_day_confidence'] = best_day.get('confidence')
            response['has_suitable_days'] = True
        else:
            response['has_suitable_days'] = False
            response['no_suitable_days_reason'] = "No suitable days for fertilizing in the next 7 days"
        
        # Convert NumPy types to Python native types for JSON serialization
        response = convert_numpy_types(response)
        
        # Use the custom JSON encoder to handle any NumPy types that might remain
        return json.loads(json.dumps(response, cls=NumpyEncoder))
        
    except Exception as e:
        print("Error predicting fertilizing suitability:", str(e))
        return jsonify({'error': str(e)}), 500

@fertilizing_bp.route('/plan', methods=['POST'])
def plan_fertilizing():
    try:
        # Get input data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON input'}), 400
        
        # Extract parameters
        location = data.get('location', 'PUTTALAM')
        rainfall_forecast = data.get('rainfall_forecast', [0.0] * 7)
        fertilizer_history = data.get('fertilizer_history', [])
        
        # Validate location
        if location not in ['PUTTALAM', 'KURUNEGALA']:
            return jsonify({'error': 'Location must be either PUTTALAM or KURUNEGALA'}), 400
        
        # Validate rainfall forecast
        if not isinstance(rainfall_forecast, list):
            return jsonify({'error': 'rainfall_forecast must be a list of values'}), 400
        
        # Ensure we have 7 days of forecast
        if len(rainfall_forecast) < 7:
            rainfall_forecast = rainfall_forecast + [0.0] * (7 - len(rainfall_forecast))
        elif len(rainfall_forecast) > 7:
            rainfall_forecast = rainfall_forecast[:7]
        
        # Convert to float
        rainfall_forecast = [float(r) for r in rainfall_forecast]
        
        # Validate fertilizer history
        if not isinstance(fertilizer_history, list):
            return jsonify({'error': 'fertilizer_history must be a list of applications'}), 400
        
        # Get fertilizer planning recommendation
        recommendation = fertilizer_planning_api(
            fertilizer_history, location, rainfall_forecast
        )
        
        # Add request info to response
        current_time = datetime.now()
        response = {
            'location': location,
            'forecast_start_date': current_time.strftime('%Y-%m-%d'),
            'current_time': current_time.strftime('%H:%M:%S'),
            'is_after_six_pm': current_time.hour >= 18,
            'recommendation': recommendation
        }
        
        # Convert NumPy types and return
        response = convert_numpy_types(response)
        return json.loads(json.dumps(response, cls=NumpyEncoder))
        
    except Exception as e:
        print("Error in fertilizer planning:", str(e))
        return jsonify({'error': str(e)}), 500

@fertilizing_bp.route('/today', methods=['POST'])
def today_fertilizing():
    try:
        # Get input data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid or missing JSON input'}), 400
        
        # Extract parameters
        location = data.get('location', 'PUTTALAM')
        rainfall = data.get('rainfall', 0.0)
        
        # Validate location
        if location not in ['PUTTALAM', 'KURUNEGALA']:
            return jsonify({'error': 'Location must be either PUTTALAM or KURUNEGALA'}), 400
        
        # Validate rainfall
        try:
            rainfall = float(rainfall)
        except ValueError:
            return jsonify({'error': 'Rainfall must be a number'}), 400
        
        # Get recommendation for today
        recommendation = predict_today_fertilizing_suitability(location, rainfall)
        
        # Check if it's after 6 PM
        current_time = datetime.now()
        is_after_six_pm = current_time.hour >= 18
        
        # Add location and time info to response
        response = {
            'location': location,
            'today': current_time.strftime('%Y-%m-%d'),
            'current_time': current_time.strftime('%H:%M:%S'),
            'is_after_six_pm': is_after_six_pm,
            'rainfall': rainfall,
            'suitable_for_fertilizing': recommendation.get('suitable_for_fertilizing', False) and not is_after_six_pm,
            'confidence': recommendation.get('confidence', 0),
            'recommendation': recommendation.get('recommendation', 'Unknown')
        }
        
        # If it's after 6 PM, modify the recommendation
        if is_after_six_pm and response['suitable_for_fertilizing']:
            response['suitable_for_fertilizing'] = False
            response['recommendation'] = "Too late for fertilizing today, check tomorrow's forecast"
        
        # Convert NumPy types to Python native types for JSON serialization
        response = convert_numpy_types(response)
        
        # Return response
        return json.loads(json.dumps(response, cls=NumpyEncoder))
        
    except Exception as e:
        print("Error predicting today's fertilizing suitability:", str(e))
        return jsonify({'error': str(e)}), 500