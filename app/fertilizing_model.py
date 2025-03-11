import joblib
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# Default feature names
DEFAULT_FEATURE_NAMES = ['Rainfall (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 
                         'Location_KURUNEGALA', 'Location_PUTTALAM']

# Default temperature values by location
DEFAULT_TEMPS = {
    'PUTTALAM': {'min': 24.5, 'max': 32.7},
    'KURUNEGALA': {'min': 24.8, 'max': 32.5}
}

def create_simple_fertilizer_model():
    """Create a simple rule-based fertilizer model without training data"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create synthetic training data based on known rules
    # Rainfall values: 0, 5, 10, 15, 20, 30, 40, 50
    rainfall_values = [0, 0, 0, 5, 5, 5, 10, 10, 15, 20, 30, 40, 50]
    
    # Temperature values: not very important for this simplified model
    min_temp_values = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
    max_temp_values = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
    
    # Location values (alternate between PUTTALAM and KURUNEGALA)
    location_puttalam = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    location_kurunegala = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Target values: 1 for suitable (low rainfall), 0 for unsuitable (high rainfall)
    # Simple rule: rainfall <= 10mm is suitable, > 10mm is unsuitable
    target_values = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    
    # Create features array
    X = np.column_stack([
        rainfall_values, 
        min_temp_values, 
        max_temp_values, 
        location_kurunegala, 
        location_puttalam
    ])
    
    # Train model on synthetic data
    model.fit(X, target_values)
    
    return model

def load_or_create_fertilizer_model():
    """Load existing model or create a new one if it doesn't exist"""
    model_path = 'models/betel_fertilizer_suitability_model.pkl'
    
    if os.path.exists(model_path):
        print("Loading existing fertilizer model")
        try:
            return joblib.load(model_path)
        except:
            print("Error loading model, creating new one")
    
    print("Creating new fertilizer model")
    model = create_simple_fertilizer_model()
    
    # Save model
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Warning: Could not save model: {str(e)}")
    
    return model

def generate_recommendation_text(is_suitable, rainfall, confidence):
    """Generate text recommendation based on prediction and rainfall"""
    if is_suitable:
        if rainfall == 0:
            return "Ideal day for fertilizing (dry conditions)"
        elif rainfall < 5:
            return "Good day for fertilizing (light rain)"
        else:
            return "Suitable for fertilizing but monitor rain"
    else:
        if rainfall > 20:
            return "Not suitable - too much rain"
        elif rainfall > 10:
            return "Not suitable - moderate rain expected"
        else:
            return "Not recommended for fertilizing today"

def predict_7day_fertilizing_suitability(location, rainfall_forecast):
    """
    Predict fertilizing suitability for a 7-day forecast
    
    Args:
        location (str): Location/district (PUTTALAM or KURUNEGALA)
        rainfall_forecast (list): 7-day rainfall forecast in mm
    
    Returns:
        list: List of dictionaries with recommendations for each day
    """
    try:
        # Load or create the model
        model = load_or_create_fertilizer_model()
        
        # Use default temperatures for the location
        avg_min_temp = DEFAULT_TEMPS.get(location, DEFAULT_TEMPS['PUTTALAM'])['min']
        avg_max_temp = DEFAULT_TEMPS.get(location, DEFAULT_TEMPS['PUTTALAM'])['max']
        
        # Create 7-day forecast dataframe
        days = []
        
        # For each day in the forecast
        for day_idx, rainfall in enumerate(rainfall_forecast):
            today = datetime.now() + timedelta(days=day_idx)
            
            # Create feature row - order must match DEFAULT_FEATURE_NAMES
            features = {
                'Rainfall (mm)': rainfall,
                'Min Temp (°C)': avg_min_temp,
                'Max Temp (°C)': avg_max_temp,
                'Location_KURUNEGALA': 1 if location == 'KURUNEGALA' else 0,
                'Location_PUTTALAM': 1 if location == 'PUTTALAM' else 0
            }
            
            # Add day information
            day_info = {
                'date': today.strftime('%Y-%m-%d'),
                'day_name': today.strftime('%A'),
                'rainfall': rainfall
            }
            
            # Combine all info
            days.append({**day_info, **features})
        
        # Create a numpy array for prediction
        # Order must match DEFAULT_FEATURE_NAMES
        X_predict = np.array([
            [
                day['Rainfall (mm)'], 
                day['Min Temp (°C)'], 
                day['Max Temp (°C)'], 
                day['Location_KURUNEGALA'], 
                day['Location_PUTTALAM']
            ] 
            for day in days
        ])
        
        # Make predictions
        predictions = model.predict(X_predict)
        probabilities = model.predict_proba(X_predict)
        
        # Create results
        results = []
        for i, day in enumerate(days):
            # Default confidence is 95% for rainfall < 5mm, 85% for 5-10mm, and lower as rainfall increases
            # Use model probability when available
            if len(probabilities[i]) > 1:
                confidence = probabilities[i][1] * 100  # Probability of class 1 (suitable)
            else:
                # Fallback if model probabilities are not available
                if day['rainfall'] < 5:
                    confidence = 95.0
                elif day['rainfall'] < 10:
                    confidence = 85.0
                elif day['rainfall'] < 15:
                    confidence = 70.0
                elif day['rainfall'] < 20:
                    confidence = 50.0
                else:
                    confidence = 30.0
            
            is_suitable = bool(predictions[i])
            
            # Simple rule override: heavy rain is never suitable
            if day['rainfall'] > 20:
                is_suitable = False
                confidence = min(confidence, 30.0)
            
            # Simple rule override: no/minimal rain is always suitable
            if day['rainfall'] < 3:
                is_suitable = True
                confidence = max(confidence, 90.0)
            
            results.append({
                'date': day['date'],
                'day_name': day['day_name'],
                'rainfall': float(day['rainfall']),
                'suitable_for_fertilizing': is_suitable,
                'confidence': float(confidence),
                'recommendation': generate_recommendation_text(is_suitable, day['rainfall'], confidence)
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