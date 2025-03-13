import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Default temperature values for Sri Lanka
DEFAULT_MIN_TEMP = 24.0
DEFAULT_MAX_TEMP = 32.0

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

# Sinhala messages for recommendations
SINHALA_RECOMMENDATIONS = {
    "dry": "පොහොර යෙදීමට ඉතාම සුදුසු දිනයකි (වියළි කාලගුණය)",
    "light_rain": "පොහොර යෙදීමට හොඳ දිනයකි (සුළු වැසි සහිත)",
    "monitor_rain": "පොහොර යෙදීමට සුදුසුයි නමුත් වැසි තත්ත්වය නිරීක්ෂණය කරන්න",
    "too_much_rain": "පොහොර යෙදීමට සුදුසු නැත - අධික වැසි",
    "moderate_rain": "පොහොර යෙදීමට සුදුසු නැත - මධ්‍යම වැසි",
    "not_recommended": "අද පොහොර යෙදීමට නිර්දේශ නොකරයි",
    "too_late": "අද පොහොර යෙදීමට ප්‍රමාද වැඩිය, හෙට උදේ බලන්න"
}

def load_fertilizer_model():
    """Load the trained betel fertilizer suitability model"""
    try:
        return joblib.load('models/betel_fertilizer_suitability_model.pkl')
    except:
        print("Could not load model, using rule-based fallback")
        return None

def get_feature_names():
    """Get the feature names used in the model"""
    try:
        return joblib.load('models/fertilizer_feature_names.pkl')
    except:
        # Default feature names if file doesn't exist
        return ['Rainfall (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 
                'Location_KURUNEGALA', 'Location_PUTTALAM']

def generate_recommendation_text(is_suitable, rainfall, confidence):
    """Generate recommendation text based on prediction and rainfall in Sinhala"""
    if is_suitable:
        if rainfall == 0:
            return SINHALA_RECOMMENDATIONS["dry"]
        elif rainfall < 5:
            return SINHALA_RECOMMENDATIONS["light_rain"]
        else:
            return SINHALA_RECOMMENDATIONS["monitor_rain"]
    else:
        if rainfall > 20:
            return SINHALA_RECOMMENDATIONS["too_much_rain"]
        elif rainfall > 10:
            return SINHALA_RECOMMENDATIONS["moderate_rain"]
        else:
            return SINHALA_RECOMMENDATIONS["not_recommended"]

def rule_based_prediction(rainfall):
    """Simple rule-based model to determine fertilizing suitability based on rainfall"""
    is_suitable = rainfall <= 10  # Suitable if 10mm or less rain
    confidence = 100 - (rainfall * 5) if rainfall <= 20 else 0
    confidence = max(0, min(100, confidence))  # Ensure between 0-100
    return is_suitable, confidence

def predict_7day_fertilizing_suitability(location, rainfall_forecast):
    """
    Predict fertilizing suitability for a 7-day forecast without requiring the Weather Data.xlsx file
    
    Args:
        location (str): Location/district (PUTTALAM or KURUNEGALA)
        rainfall_forecast (list): 7-day rainfall forecast in mm
    
    Returns:
        list: List of dictionaries with recommendations for each day
    """
    try:
        # Try to load the model, use rule-based approach if not available
        model = load_fertilizer_model()
        feature_names = get_feature_names()
        use_model = model is not None
        
        # Create 7-day forecast dataframe
        days = []
        
        # For each day in the forecast
        for day_idx, rainfall in enumerate(rainfall_forecast):
            today = datetime.now() + timedelta(days=day_idx)
            
            # Create feature row
            features = {
                'Rainfall (mm)': rainfall,
                'Min Temp (°C)': DEFAULT_MIN_TEMP,
                'Max Temp (°C)': DEFAULT_MAX_TEMP,
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
                'rainfall': rainfall
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
            forecast_df['suitable_for_fertilizing'] = predictions
            forecast_df['confidence'] = [prob[1] * 100 for prob in probabilities]  # Confidence for "suitable" class
        else:
            # Rule-based approach
            rule_results = [rule_based_prediction(rainfall) for rainfall in rainfall_forecast]
            forecast_df['suitable_for_fertilizing'] = [result[0] for result in rule_results]
            forecast_df['confidence'] = [result[1] for result in rule_results]
        
        # Create recommendation text
        forecast_df['recommendation'] = forecast_df.apply(
            lambda row: generate_recommendation_text(
                row['suitable_for_fertilizing'], row['rainfall'], row['confidence']
            ),
            axis=1
        )
        
        # Convert to list of dictionaries
        results = []
        for _, row in forecast_df.iterrows():
            # Get the Sinhala day name
            english_day_name = row['day_name']
            sinhala_day_name = SINHALA_DAY_NAMES.get(english_day_name, english_day_name)
            
            results.append({
                'date': row['date'],
                'day_name': english_day_name,
                'day_name_sinhala': sinhala_day_name,  # Add Sinhala day name
                'rainfall': float(row['rainfall']),
                'suitable_for_fertilizing': bool(row['suitable_for_fertilizing']),
                'confidence': float(row['confidence']),
                'recommendation': row['recommendation']
            })
        
        # Find best day
        suitable_days = [day for day in results if day['suitable_for_fertilizing']]
        if suitable_days:
            best_day = max(suitable_days, key=lambda x: x['confidence'])
            best_day_idx = results.index(best_day)
            results[best_day_idx]['is_best_day'] = True
        
        return results
    
    except Exception as e:
        print(f"Error predicting fertilizing suitability: {str(e)}")
        # Return error message
        return [{'error': str(e)}]

def predict_today_fertilizing_suitability(location, rainfall):
    """
    Predict fertilizing suitability for today based on rainfall data only.
    Works without requiring the Weather Data.xlsx file.
    
    Args:
        location (str): Location/district (PUTTALAM or KURUNEGALA)
        rainfall (float): Today's rainfall in mm
    
    Returns:
        dict: Recommendation for today
    """
    today = datetime.now()
    
    try:
        # Try to use the existing model first
        try:
            model = load_fertilizer_model()
            
            if model:
                feature_names = get_feature_names()
                
                # Create features dictionary
                features = {
                    'Rainfall (mm)': rainfall,
                    'Min Temp (°C)': DEFAULT_MIN_TEMP,
                    'Max Temp (°C)': DEFAULT_MAX_TEMP,
                    'Location_KURUNEGALA': 0,
                    'Location_PUTTALAM': 0
                }
                
                # Set the correct location column to 1
                location_col = f'Location_{location}'
                if location_col in features:
                    features[location_col] = 1
                
                # Convert to DataFrame
                df = pd.DataFrame([features])
                
                # Ensure all required columns are present
                for col in feature_names:
                    if col not in df.columns:
                        df[col] = 0
                
                # Reorder columns to match training data
                input_features = df[feature_names]
                
                # Make prediction
                prediction = model.predict(input_features)[0]
                probabilities = model.predict_proba(input_features)[0]
                confidence = probabilities[1] * 100  # Confidence for "suitable" class
            else:
                raise Exception("Model not available")
                
        except Exception as e:
            print(f"Model prediction failed, using rule-based approach: {e}")
            # Fall back to rule-based approach if model isn't available
            prediction, confidence = rule_based_prediction(rainfall)
        
        # Generate recommendation text
        recommendation = generate_recommendation_text(bool(prediction), rainfall, confidence)
        
        # Get Sinhala day name
        english_day_name = today.strftime('%A')
        sinhala_day_name = SINHALA_DAY_NAMES.get(english_day_name, english_day_name)
        
        # Create response
        return {
            'date': today.strftime('%Y-%m-%d'),
            'day_name': english_day_name,
            'day_name_sinhala': sinhala_day_name,  # Add Sinhala day name
            'rainfall': float(rainfall),
            'suitable_for_fertilizing': bool(prediction),
            'confidence': float(confidence),
            'recommendation': recommendation
        }
    
    except Exception as e:
        print(f"Error predicting fertilizing suitability: {str(e)}")
        
        # Get Sinhala day name even in error case
        english_day_name = today.strftime('%A')
        sinhala_day_name = SINHALA_DAY_NAMES.get(english_day_name, english_day_name)
        
        return {
            'date': today.strftime('%Y-%m-%d'),
            'day_name': english_day_name,
            'day_name_sinhala': sinhala_day_name,  # Add Sinhala day name
            'rainfall': float(rainfall),
            'suitable_for_fertilizing': False,
            'confidence': 0.0,
            'recommendation': f"දෝෂයකි: {str(e)}"  # Error in Sinhala
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