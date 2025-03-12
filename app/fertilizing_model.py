import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Path to historical data file
HISTORICAL_DATA_PATH = os.path.join('data', 'Weather Data.xlsx')

def load_fertilizer_model():
    """Load the trained betel fertilizer suitability model"""
    return joblib.load('models/betel_fertilizer_suitability_model.pkl')

def train_fertilizer_model():
    """Train and save the fertilizer suitability model if it doesn't exist"""
    model_path = 'models/betel_fertilizer_suitability_model.pkl'
    
    if os.path.exists(model_path):
        print("Using existing fertilizer model")
        return load_fertilizer_model()
    
    print("Training new fertilizer model...")
    
    try:
        # Load data
        df = pd.read_excel(HISTORICAL_DATA_PATH)
        
        # Filter for only Puttalam and Kurunegala
        df = df[df['Location'].isin(['PUTTALAM', 'KURUNEGALA'])]
        
        # Add binary target for fertilization
        df['Can_Fertilize'] = (df['Fertilizing Recommendation'] == 'Recommended').astype(int)
        
        # Features for model
        features = df[['Rainfall (mm)', 'Min Temp (°C)', 'Max Temp (°C)']]
        
        # Add location as one-hot encoded feature
        location_dummies = pd.get_dummies(df['Location'], prefix='Location')
        features = pd.concat([features, location_dummies], axis=1)
        
        target = df['Can_Fertilize']
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, target)
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # Save feature names for later reference
        joblib.dump(features.columns.tolist(), 'models/fertilizer_feature_names.pkl')
        
        print(f"Fertilizer model trained and saved to {model_path}")
        return model
    
    except Exception as e:
        print(f"Error training fertilizer model: {str(e)}")
        # Return a dummy model if training fails
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()

def get_feature_names():
    """Get the feature names used in the model"""
    try:
        return joblib.load('models/fertilizer_feature_names.pkl')
    except:
        # Default feature names if file doesn't exist
        return ['Rainfall (mm)', 'Min Temp (°C)', 'Max Temp (°C)', 
                'Location_KURUNEGALA', 'Location_PUTTALAM']

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
        # Load or train the model
        model = train_fertilizer_model()
        feature_names = get_feature_names()
        
        # Get sample data for temperature (since we're focusing mainly on rainfall)
        df = pd.read_excel(HISTORICAL_DATA_PATH)
        loc_data = df[df['Location'] == location]
        avg_min_temp = loc_data['Min Temp (°C)'].mean()
        avg_max_temp = loc_data['Max Temp (°C)'].mean()
        
        # Create 7-day forecast dataframe
        days = []
        
        # For each day in the forecast
        for day_idx, rainfall in enumerate(rainfall_forecast):
            today = datetime.now() + timedelta(days=day_idx)
            
            # Create feature row
            features = {
                'Rainfall (mm)': rainfall,
                'Min Temp (°C)': avg_min_temp,
                'Max Temp (°C)': avg_max_temp,
                # Set all location columns to 0 initially
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
            results.append({
                'date': row['date'],
                'day_name': row['day_name'],
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
            feature_names = get_feature_names()
            
            # Default temperatures (reasonable averages for Sri Lanka)
            default_min_temp = 24.0  # Default min temp 
            default_max_temp = 32.0  # Default max temp 
            
            # Create features dictionary
            features = {
                'Rainfall (mm)': rainfall,
                'Min Temp (°C)': default_min_temp,
                'Max Temp (°C)': default_max_temp,
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
            
        except Exception as e:
            print(f"Model prediction failed, using rule-based approach: {e}")
            # Fall back to rule-based approach if model isn't available
            prediction = rainfall <= 10  # True if rainfall is 10mm or less
            confidence = 100 - (rainfall * 5) if rainfall <= 20 else 0  # Simple rule
            confidence = max(0, min(100, confidence))  # Ensure between 0-100
        
        # Generate recommendation text
        recommendation = generate_recommendation_text(bool(prediction), rainfall, confidence)
        
        # Create response
        return {
            'date': today.strftime('%Y-%m-%d'),
            'day_name': today.strftime('%A'),
            'rainfall': float(rainfall),
            'suitable_for_fertilizing': bool(prediction),
            'confidence': float(confidence),
            'recommendation': recommendation
        }
    
    except Exception as e:
        print(f"Error predicting fertilizing suitability: {str(e)}")
        return {
            'date': today.strftime('%Y-%m-%d'),
            'day_name': today.strftime('%A'),
            'rainfall': float(rainfall),
            'suitable_for_fertilizing': False,
            'confidence': 0.0,
            'recommendation': f"Error predicting suitability: {str(e)}"
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