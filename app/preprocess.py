import joblib
import pandas as pd
import numpy as np

# Load the preprocessor
preprocessor = joblib.load('models/preprocessor_P.pkl')

def preprocess_input_data(input_data):
    """Preprocess input data to match the training format."""
    # Convert dates to datetime
    input_data['Last Harvest Date'] = pd.to_datetime(input_data['Last Harvest Date'], format='%m/%d/%Y')
    input_data['Expected Harvest Date'] = pd.to_datetime(input_data['Expected Harvest Date'], format='%m/%d/%Y')
    
    # Calculate derived features
    input_data['Days Between Harvests'] = (input_data['Expected Harvest Date'] - input_data['Last Harvest Date']).dt.days
    input_data['Avg Rainfall'] = input_data['Rainfall Seq (mm)'].apply(np.mean)
    input_data['Avg Min Temp'] = input_data['Min Temp Seq (°C)'].apply(np.mean)
    input_data['Avg Max Temp'] = input_data['Max Temp Seq (°C)'].apply(np.mean)
    input_data['Harvest Month'] = input_data['Expected Harvest Date'].dt.month
    input_data['Planting Density'] = input_data['Planted Sticks'] / input_data['Land Size (acres)']
    input_data['Temp Range'] = input_data['Avg Max Temp'] - input_data['Avg Min Temp']
    input_data['Avg Daily Rainfall'] = input_data['Avg Rainfall'] / input_data['Days Between Harvests']

    # Drop unnecessary columns
    input_data = input_data.drop(
        ['Last Harvest Date', 'Expected Harvest Date', 'Rainfall Seq (mm)', 'Min Temp Seq (°C)', 'Max Temp Seq (°C)'], 
        axis=1)
    
    # Apply the preprocessor
    processed_data = preprocessor.transform(input_data)

    return processed_data

