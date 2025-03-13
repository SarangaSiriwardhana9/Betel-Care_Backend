from flask import Blueprint, request, jsonify
import json
import numpy as np
from datetime import datetime
from app.protection_model import predict_7day_protection_needs, predict_today_protection_needs, convert_numpy_types

protection_bp = Blueprint('protection', __name__)

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

@protection_bp.route('/predict', methods=['POST'])
def predict_protection():
    try:
        # Get input data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'දත්ත ආදානය අවලංගුයි හෝ නොමැත'}), 400
        
        # Extract parameters with default values
        location = data.get('location', 'PUTTALAM')  # Default to PUTTALAM if not specified
        rainfall_forecast = data.get('rainfall_forecast', [0.0] * 7)  # Default to 7 dry days
        min_temp_forecast = data.get('min_temp_forecast')
        max_temp_forecast = data.get('max_temp_forecast')
        
        # Validate location
        if location not in ['PUTTALAM', 'KURUNEGALA']:
            return jsonify({'error': 'ස්ථානය PUTTALAM හෝ KURUNEGALA විය යුතුය'}), 400
        
        # Validate rainfall forecast
        if not isinstance(rainfall_forecast, list):
            return jsonify({'error': 'වර්ෂාපතන පුරෝකථනය අගයන් ලැයිස්තුවක් විය යුතුය'}), 400
        
        # Ensure we have 7 days of forecast
        if len(rainfall_forecast) < 7:
            # If less than 7 days provided, pad with zeros
            rainfall_forecast = rainfall_forecast + [0.0] * (7 - len(rainfall_forecast))
        elif len(rainfall_forecast) > 7:
            # If more than 7 days provided, truncate
            rainfall_forecast = rainfall_forecast[:7]
        
        # Convert to float (in case they're strings in the JSON)
        rainfall_forecast = [float(r) for r in rainfall_forecast]
        
        # Process temperature forecasts if provided
        if min_temp_forecast is not None:
            if not isinstance(min_temp_forecast, list):
                return jsonify({'error': 'අවම උෂ්ණත්ව පුරෝකථනය අගයන් ලැයිස්තුවක් විය යුතුය'}), 400
            
            # Ensure we have 7 days
            if len(min_temp_forecast) < 7:
                min_temp_forecast = min_temp_forecast + [24.0] * (7 - len(min_temp_forecast))
            elif len(min_temp_forecast) > 7:
                min_temp_forecast = min_temp_forecast[:7]
            
            min_temp_forecast = [float(t) for t in min_temp_forecast]
        
        if max_temp_forecast is not None:
            if not isinstance(max_temp_forecast, list):
                return jsonify({'error': 'උපරිම උෂ්ණත්ව පුරෝකථනය අගයන් ලැයිස්තුවක් විය යුතුය'}), 400
            
            # Ensure we have 7 days
            if len(max_temp_forecast) < 7:
                max_temp_forecast = max_temp_forecast + [32.0] * (7 - len(max_temp_forecast))
            elif len(max_temp_forecast) > 7:
                max_temp_forecast = max_temp_forecast[:7]
            
            max_temp_forecast = [float(t) for t in max_temp_forecast]
        
        # Get protection recommendations
        recommendations = predict_7day_protection_needs(
            location, 
            rainfall_forecast, 
            min_temp_forecast, 
            max_temp_forecast
        )
        
        # Add location and current time information to response
        current_time = datetime.now()
        
        response = {
            'location': location,
            'forecast_start_date': current_time.strftime('%Y-%m-%d'),
            'current_time': current_time.strftime('%H:%M:%S'),
            'daily_recommendations': recommendations
        }
        
        # Find days requiring special protection
        protection_days = [day for day in recommendations 
                           if day.get('protection_type', 0) > 0 
                           and not 'error' in day]
        
        if protection_days:
            response['has_protection_days'] = True
            
            # Group protection days by type
            drought_days = [day for day in protection_days if day.get('protection_type') == 1]
            excess_rain_days = [day for day in protection_days if day.get('protection_type') == 2]
            
            # Add summary info
            response['drought_protection_days'] = len(drought_days)
            response['excess_rain_protection_days'] = len(excess_rain_days)
            
            # Find best day for each protection type
            if drought_days:
                best_drought_day = max(drought_days, key=lambda x: x.get('drought_protection_confidence', 0))
                response['best_drought_protection_day'] = best_drought_day.get('date')
            
            if excess_rain_days:
                best_excess_rain_day = max(excess_rain_days, key=lambda x: x.get('excess_rain_protection_confidence', 0))
                response['best_excess_rain_protection_day'] = best_excess_rain_day.get('date')
        else:
            response['has_protection_days'] = False
            response['no_protection_days_reason'] = "ඉදිරි දින 7 තුළ විශේෂ ආරක්ෂණයක් අවශ්‍ය නොවේ"
        
        # Convert NumPy types to Python native types for JSON serialization
        response = convert_numpy_types(response)
        
        # Use the custom JSON encoder to handle any NumPy types that might remain
        return json.loads(json.dumps(response, cls=NumpyEncoder))
        
    except Exception as e:
        print("Error predicting protection needs:", str(e))
        return jsonify({'error': f'ආරක්ෂණ අවශ්‍යතා පුරෝකථනය කිරීමේ දෝෂයකි: {str(e)}'}), 500

@protection_bp.route('/today', methods=['POST'])
def today_protection():
    try:
        # Get input data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'දත්ත ආදානය අවලංගුයි හෝ නොමැත'}), 400
        
        # Extract parameters
        location = data.get('location', 'PUTTALAM')
        rainfall = data.get('rainfall', 0.0)
        min_temp = data.get('min_temp')
        max_temp = data.get('max_temp')
        
        # Validate location
        if location not in ['PUTTALAM', 'KURUNEGALA']:
            return jsonify({'error': 'ස්ථානය PUTTALAM හෝ KURUNEGALA විය යුතුය'}), 400
        
        # Validate rainfall
        try:
            rainfall = float(rainfall)
        except ValueError:
            return jsonify({'error': 'වර්ෂාපතනය සංඛ්‍යාවක් විය යුතුය'}), 400
        
        # Validate temperatures if provided
        if min_temp is not None:
            try:
                min_temp = float(min_temp)
            except ValueError:
                return jsonify({'error': 'අවම උෂ්ණත්වය සංඛ්‍යාවක් විය යුතුය'}), 400
        
        if max_temp is not None:
            try:
                max_temp = float(max_temp)
            except ValueError:
                return jsonify({'error': 'උපරිම උෂ්ණත්වය සංඛ්‍යාවක් විය යුතුය'}), 400
        
        # Get recommendation for today
        recommendation = predict_today_protection_needs(location, rainfall, min_temp, max_temp)
        
        # Add location and time info to response
        current_time = datetime.now()
        response = {
            'location': location,
            'today': current_time.strftime('%Y-%m-%d'),
            'current_time': current_time.strftime('%H:%M:%S'),
            'recommendation': recommendation
        }
        
        # Convert NumPy types to Python native types for JSON serialization
        response = convert_numpy_types(response)
        
        # Return response
        return json.loads(json.dumps(response, cls=NumpyEncoder))
        
    except Exception as e:
        print("Error predicting today's protection needs:", str(e))
        return jsonify({'error': f'අද ආරක්ෂණ අවශ්‍යතා පුරෝකථනය කිරීමේ දෝෂයකි: {str(e)}'}), 500