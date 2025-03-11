import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Path to historical data file
HISTORICAL_DATA_PATH = os.path.join('data', 'Weather Data.xlsx')

def load_watering_model():
    """Load the trained betel watering model"""
    return joblib.load('models/betel_watering_model.pkl')

def calculate_consecutive_dry_days(location):
    """Calculate consecutive dry days for a location based on historical data"""
    try:
        df = pd.read_excel(HISTORICAL_DATA_PATH)
        # Filter for only the specified location
        df = df[df['Location'] == location]
        
        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date in descending order (newest first)
        df = df.sort_values('Date', ascending=False)
        
        # Count consecutive dry days from the most recent record
        dry_days = 0
        for _, row in df.iterrows():
            if row['Rainfall (mm)'] == 0:
                dry_days += 1
            else:
                break
        
        return dry_days
    except Exception as e:
        print(f"Warning: Could not calculate consecutive dry days: {str(e)}")
        return 0

def prepare_features(location, rainfall, min_temp, max_temp, crop_stage, month):
    """Prepare feature dataframe for watering prediction"""
    # Calculate consecutive dry days from historical data
    consecutive_dry_days = calculate_consecutive_dry_days(location)
    
    # Create a DataFrame with features
    new_data = pd.DataFrame({
        'Rainfall (mm)': [float(rainfall)],
        'Min Temp (°C)': [float(min_temp)],
        'Max Temp (°C)': [float(max_temp)],
        'ConsecutiveDryDays': [int(consecutive_dry_days)],
        'CropStageValue': [int(crop_stage)],
        'Month': [int(month)]
    })
    
    # Add temperature category
    if max_temp < 28:
        temp_cat = 'cool'
    elif max_temp < 32:
        temp_cat = 'moderate'
    else:
        temp_cat = 'hot'
        
    # Add rainfall category
    if rainfall == 0:
        rain_cat = 'none'
    elif rainfall < 5:
        rain_cat = 'light'
    elif rainfall < 20:
        rain_cat = 'moderate'
    else:
        rain_cat = 'heavy'
        
    # One-hot encode these categories
    for cat in ['cool', 'moderate', 'hot']:
        new_data[f'Temp_{cat}'] = 1 if temp_cat == cat else 0
        
    for cat in ['none', 'light', 'moderate', 'heavy']:
        new_data[f'Rain_{cat}'] = 1 if rain_cat == cat else 0
    
    # Get the feature names from the model to ensure compatibility
    model = load_watering_model()
    if hasattr(model, 'feature_names_in_'):
        # Ensure we have the same features in the same order as the model expects
        required_features = model.feature_names_in_
        
        # Add any missing columns with zeros
        for feature in required_features:
            if feature not in new_data.columns:
                new_data[feature] = 0
        
        # Reorder columns to match the model's expected order
        new_data = new_data[required_features]
    
    return new_data

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

def get_watering_recommendation(model, features):
    """Generate watering recommendation based on model prediction"""
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Map prediction to recommendation
    if prediction == 0:
        recommendation = "No watering needed"
        water_amount = 0
    elif prediction == 1:
        recommendation = "Water once today"
        # Base amount on temperature
        if features['Max Temp (°C)'].values[0] > 32:
            water_amount = 5  # Higher for hot days
        else:
            water_amount = 4  # Normal amount
    else:  # prediction == 2
        recommendation = "Water twice today"
        water_amount = 8  # Higher amount for twice watering
    
    confidence = float(probabilities[prediction] * 100)
    
    # Extract consecutive dry days, ensuring it's a Python integer
    consecutive_dry_days = 0
    if 'ConsecutiveDryDays' in features:
        consecutive_dry_days = int(features['ConsecutiveDryDays'].values[0])
    
    # Create the recommendation dictionary with Python native types
    result = {
        'recommendation': recommendation,
        'water_amount': int(water_amount),
        'confidence': confidence,
        'consecutive_dry_days': consecutive_dry_days,
        'probabilities': {
            'No watering': float(probabilities[0] * 100),
            'Water once': float(probabilities[1] * 100),
            'Water twice': float(probabilities[2] * 100)
        }
    }
    
    # Convert any remaining NumPy types
    return convert_numpy_types(result)