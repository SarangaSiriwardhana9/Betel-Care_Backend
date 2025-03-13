import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Dictionary for Sinhala day names
SINHALA_DAY_NAMES = {
    "Monday": "සඳුදා",
    "Tuesday": "අඟහරුවාදා",
    "Wednesday": "බදාදා",
    "Thursday": "බ්‍රහස්පතින්දා",
    "Friday": "සිකුරාදා",
    "Saturday": "සෙනසුරාදා",
    "Sunday": "ඉරිදා"
}

# Default temperature values for Sri Lanka if not provided
DEFAULT_MIN_TEMP = 24.0
DEFAULT_MAX_TEMP = 32.0

def load_protection_model():
    """Load the trained betel protection model"""
    try:
        return joblib.load('models/betel_protection_model.pkl')
    except:
        print("Could not load protection model, using rule-based fallback")
        return None

def get_feature_names():
    """Get the feature names used in the model"""
    try:
        return joblib.load('models/protection_feature_names.pkl')
    except:
        # Default feature names if file doesn't exist
        return ['Rainfall (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 
                'Location_KURUNEGALA', 'Location_PUTTALAM']

def get_protection_methods(protection_type, max_temp, min_temp, rainfall):
    """
    Determine the best protection methods based on protection type and weather conditions
    Also returns Sinhala translations for recommendations
    """
    
    if protection_type == 0:
        return {
            "english": ["No special protection needed - maintain regular care"],
            "sinhala": ["විශේෂ ආරක්ෂණ විධිවිධාන අවශ්‍ය නැත - සාමාන්‍ය රැකවරණය පවත්වා ගන්න"]
        }
    
    # Drought protection methods (Type 1)
    if protection_type == 1:
        english_methods = []
        sinhala_methods = []
        
        # High temperature, low rainfall (drought conditions)
        english_methods.append("MU - Mulching: Spread organic mulch around plants to retain soil moisture")
        sinhala_methods.append("MU - මල්චිං: පසෙහි තෙතමනය රඳවා ගැනීමට ශාක වටා කාබනික මල්ච් පැතිරවීම")
        
        if max_temp >= 34:
            english_methods.append("SH - Shading: Use shade nets during peak sun hours (10am-3pm)")
            sinhala_methods.append("SH - සෙවන: උच්ච හිරු පැය තුළ (පෙ.ව 10 - ප.ව 3) සෙවන දැල් භාවිතා කරන්න")
        
        english_methods.append("FW - Frequent Watering: Water plants early morning or late evening")
        sinhala_methods.append("FW - නිතර ජලය දැමීම: පැළෑටි වලට උදෑසන හෝ සවස් යාමයේ ජලය දමන්න")
        
        if max_temp - min_temp > 10:  # Large temperature variation
            english_methods.append("SE - Soil Enrichment: Add compost to improve water retention")
            sinhala_methods.append("SE - පස පොෂණය: ජලය රඳවා ගැනීම වැඩි දියුණු කිරීමට කොම්පෝස්ට් එකතු කරන්න")
            
        return {
            "english": english_methods,
            "sinhala": sinhala_methods
        }
    
    # Excess rain protection methods (Type 2)
    if protection_type == 2:
        english_methods = []
        sinhala_methods = []
        
        if rainfall >= 30:
            english_methods.append("DS - Drainage System: Create channels to divert excess water")
            sinhala_methods.append("DS - ජලාපවහන පද්ධතිය: අතිරික්ත ජලය ඉවත් කිරීමට කාණු සාදන්න")
            
            english_methods.append("CP - Cover Plants: Use polythene sheets as temporary rain protection")
            sinhala_methods.append("CP - පැළ ආවරණය: තාවකාලික වැසි ආරක්ෂාව සඳහා පොලිතීන් ෂීට් භාවිතා කරන්න")
        else:
            english_methods.append("DS - Drainage System: Ensure proper drainage around plants")
            sinhala_methods.append("DS - ජලාපවහන පද්ධතිය: පැළ වටා නිසි ජලාපවහනය සහතික කරන්න")
        
        english_methods.append("SP - Stake Plants: Tie plants to stakes to prevent damage from wind and water")
        sinhala_methods.append("SP - පැළ රඳවන: සුළඟ සහ ජලයෙන් හානි වීම වැළැක්වීමට පැළ කූරු වලට බැඳ තබන්න")
        
        if rainfall >= 25:
            english_methods.append("AF - Avoid Fertilizing: Postpone fertilization until drier conditions")
            sinhala_methods.append("AF - පොහොර දැමීමෙන් වළකින්න: වියළි තත්ත්වයන් එනතුරු පොහොර දැමීම කල් දමන්න")
            
            english_methods.append("PI - Pest Inspection: Check for increased pest activity after rain")
            sinhala_methods.append("PI - පළිබෝධ පරීක්ෂාව: වැස්සෙන් පසු වැඩි වූ පළිබෝධ ක්‍රියාකාරකම් පරීක්ෂා කරන්න")
            
        return {
            "english": english_methods,
            "sinhala": sinhala_methods
        }
    
    return {
        "english": ["Standard care: Monitor plants regularly"],
        "sinhala": ["සාමාන්‍ය රැකවරණය: නිතිපතා පැළෑටි නිරීක්ෂණය කරන්න"]
    }

def rule_based_protection(rainfall, min_temp, max_temp):
    """Simple rule-based model to determine protection type based on weather conditions"""
    
    # Define thresholds based on betel farming best practices
    HIGH_TEMP_THRESHOLD = 33.0  # High temperature threshold in Celsius
    LOW_RAIN_THRESHOLD = 2.0    # Low rainfall threshold in mm
    HIGH_RAIN_THRESHOLD = 20.0  # High rainfall threshold in mm
    
    # Excess rain conditions (highest priority)
    if rainfall >= HIGH_RAIN_THRESHOLD:
        protection_type = 2  # Excess Rain Protection
        confidence = min(100, rainfall * 2)  # Higher rainfall = higher confidence
    
    # Drought conditions
    elif max_temp >= HIGH_TEMP_THRESHOLD and rainfall <= LOW_RAIN_THRESHOLD:
        protection_type = 1  # Drought Protection
        confidence = min(100, (max_temp - HIGH_TEMP_THRESHOLD) * 20)  # Higher temp = higher confidence
    
    # Normal conditions
    else:
        protection_type = 0  # No Protection Needed
        confidence = 80  # Default confidence
    
    # Create probability distribution
    if protection_type == 0:
        probabilities = [confidence/100, (100-confidence)/200, (100-confidence)/200]
    elif protection_type == 1:
        probabilities = [(100-confidence)/200, confidence/100, (100-confidence)/200]
    else:  # protection_type == 2
        probabilities = [(100-confidence)/200, (100-confidence)/200, confidence/100]
    
    return protection_type, probabilities

def predict_7day_protection_needs(location, rainfall_forecast, min_temp_forecast=None, max_temp_forecast=None):
    """
    Predict protection needs for a 7-day forecast
    
    Args:
        location (str): Location/district (PUTTALAM or KURUNEGALA)
        rainfall_forecast (list): 7-day rainfall forecast in mm
        min_temp_forecast (list, optional): 7-day minimum temperature forecast in Celsius
        max_temp_forecast (list, optional): 7-day maximum temperature forecast in Celsius
    
    Returns:
        list: List of dictionaries with recommendations for each day
    """
    try:
        # Try to load the model, use rule-based approach if not available
        model = load_protection_model()
        feature_names = get_feature_names()
        use_model = model is not None
        
        # Create default temperature forecasts if not provided
        if min_temp_forecast is None:
            min_temp_forecast = [DEFAULT_MIN_TEMP] * len(rainfall_forecast)
        if max_temp_forecast is None:
            max_temp_forecast = [DEFAULT_MAX_TEMP] * len(rainfall_forecast)
        
        # Ensure all forecasts are the same length
        forecast_length = min(len(rainfall_forecast), len(min_temp_forecast), len(max_temp_forecast))
        rainfall_forecast = rainfall_forecast[:forecast_length]
        min_temp_forecast = min_temp_forecast[:forecast_length]
        max_temp_forecast = max_temp_forecast[:forecast_length]
        
        # Create 7-day forecast dataframe
        days = []
        
        # For each day in the forecast
        for day_idx in range(forecast_length):
            today = datetime.now() + timedelta(days=day_idx)
            rainfall = rainfall_forecast[day_idx]
            min_temp = min_temp_forecast[day_idx]
            max_temp = max_temp_forecast[day_idx]
            
            # Create feature row
            features = {
                'Rainfall (mm)': rainfall,
                'Min Temp (°C)': min_temp,
                'Max Temp (°C)': max_temp,
                'Location_KURUNEGALA': 0,
                'Location_PUTTALAM': 0
            }
            
            # Set the correct location column to 1
            location_col = f'Location_{location}'
            if location_col in features:
                features[location_col] = 1
            
            # Add day information
            day_info = {
                'date': today.strftime('%Y-%m-%d'),
                'day_name': today.strftime('%A'),
                'day_name_sinhala': SINHALA_DAY_NAMES.get(today.strftime('%A'), today.strftime('%A')),
                'rainfall': rainfall,
                'min_temp': min_temp,
                'max_temp': max_temp
            }
            
            # Combine all info
            days.append({**day_info, **features})
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(days)
        
        # Predict using model or rule-based approach
        if use_model:
            # Ensure all required columns are present
            for col in feature_names:
                if col not in forecast_df.columns:
                    forecast_df[col] = 0
            
            # Reorder columns to match training data
            input_features = forecast_df[feature_names]
            
            # Make predictions
            predictions = model.predict(input_features)
            probabilities = model.predict_proba(input_features)
            
            # Add predictions to forecast
            forecast_df['protection_type'] = predictions
            
            # Add confidence scores for each class
            for i, class_name in enumerate(['no_protection_confidence', 
                                         'drought_protection_confidence', 
                                         'excess_rain_protection_confidence']):
                forecast_df[class_name] = [prob[i] * 100 for prob in probabilities]
        else:
            # Rule-based approach
            rule_results = [rule_based_protection(r, mt, xt) 
                           for r, mt, xt in zip(rainfall_forecast, min_temp_forecast, max_temp_forecast)]
            
            forecast_df['protection_type'] = [result[0] for result in rule_results]
            
            # Create probability arrays
            probs = [result[1] for result in rule_results]
            forecast_df['no_protection_confidence'] = [p[0] * 100 for p in probs]
            forecast_df['drought_protection_confidence'] = [p[1] * 100 for p in probs]
            forecast_df['excess_rain_protection_confidence'] = [p[2] * 100 for p in probs]
        
        # Map protection types to descriptive labels
        protection_type_map = {
            0: 'No Protection Needed',
            1: 'Drought Protection',
            2: 'Excess Rain Protection'
        }
        
        protection_type_map_sinhala = {
            0: 'විශේෂ ආරක්ෂණයක් අවශ්‍ය නැත',
            1: 'නියඟ ආරක්ෂණය',
            2: 'අධික වැසි ආරක්ෂණය'
        }
        
        forecast_df['protection_label'] = forecast_df['protection_type'].map(protection_type_map)
        forecast_df['protection_label_sinhala'] = forecast_df['protection_type'].map(protection_type_map_sinhala)
        
        # Add protection methods based on type and weather conditions
        forecast_df['protection_methods'] = forecast_df.apply(
            lambda row: get_protection_methods(
                row['protection_type'], row['max_temp'], row['min_temp'], row['rainfall']
            ),
            axis=1
        )
        
        # Convert to list of dictionaries
        results = []
        for _, row in forecast_df.iterrows():
            results.append({
                'date': row['date'],
                'day_name': row['day_name'],
                'day_name_sinhala': row['day_name_sinhala'],
                'rainfall': float(row['rainfall']),
                'min_temp': float(row['min_temp']),
                'max_temp': float(row['max_temp']),
                'protection_type': int(row['protection_type']),
                'protection_label': row['protection_label'],
                'protection_label_sinhala': row['protection_label_sinhala'],
                'no_protection_confidence': float(row['no_protection_confidence']),
                'drought_protection_confidence': float(row['drought_protection_confidence']),
                'excess_rain_protection_confidence': float(row['excess_rain_protection_confidence']),
                'protection_methods_english': row['protection_methods']['english'],
                'protection_methods_sinhala': row['protection_methods']['sinhala']
            })
        
        # Find the day with highest confidence for its protection type
        best_days = {}
        for ptype in [0, 1, 2]:
            type_days = [day for day in results if day['protection_type'] == ptype]
            if type_days:
                if ptype == 0:
                    confidence_key = 'no_protection_confidence'
                elif ptype == 1:
                    confidence_key = 'drought_protection_confidence'
                else:
                    confidence_key = 'excess_rain_protection_confidence'
                
                best_day = max(type_days, key=lambda x: x[confidence_key])
                best_days[ptype] = best_day
        
        # Mark best days in results
        for day in results:
            ptype = day['protection_type']
            if ptype in best_days and day['date'] == best_days[ptype]['date']:
                day['is_best_day'] = True
            else:
                day['is_best_day'] = False
        
        return results
    
    except Exception as e:
        print(f"Error predicting protection needs: {str(e)}")
        # Return error message
        return [{'error': str(e)}]

def predict_today_protection_needs(location, rainfall, min_temp=None, max_temp=None):
    """
    Predict protection needs for today based on weather data.
    
    Args:
        location (str): Location/district (PUTTALAM or KURUNEGALA)
        rainfall (float): Today rainfall in mm
        min_temp (float, optional): Today minimum temperature in Celsius
        max_temp (float, optional): Today maximum temperature in Celsius
    
    Returns:
        dict: Recommendation for today
    """
    today = datetime.now()
    
    try:
        # Use default temperatures if not provided
        if min_temp is None:
            min_temp = DEFAULT_MIN_TEMP
        if max_temp is None:
            max_temp = DEFAULT_MAX_TEMP
        
        # Simply call the 7-day function with just today data
        results = predict_7day_protection_needs(
            location, 
            [rainfall],
            [min_temp],
            [max_temp]
        )
        
        if results and len(results) > 0:
            if 'error' in results[0]:
                return results[0]
                
            return results[0]
        else:
            raise Exception("No prediction results returned")
        
    except Exception as e:
        print(f"Error predicting today protection needs: {str(e)}")
        return {
            'date': today.strftime('%Y-%m-%d'),
            'day_name': today.strftime('%A'),
            'day_name_sinhala': SINHALA_DAY_NAMES.get(today.strftime('%A'), today.strftime('%A')),
            'rainfall': float(rainfall),
            'min_temp': float(min_temp if min_temp is not None else DEFAULT_MIN_TEMP),
            'max_temp': float(max_temp if max_temp is not None else DEFAULT_MAX_TEMP),
            'error': str(e)
        }

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj