from flask import  jsonify
from pydantic import BaseModel
import pandas as pd
import joblib


# Paths to the models, preprocessors, and encoders
PRICE_MODEL_PATH = "./models/kavindi/price/betal_price_prediction_model.pkl"
PRICE_ENCODER_PATH = "./models/kavindi/price/betel_label_encoders.pkl"
PRICE_PREPROCESSOR_PATH = "./models/kavindi/price/betel_preprocessing.pkl"

DEMAND_LOCATION_MODEL_PATH = "./models/kavindi/demand/betel_location_prediction_model.pkl"
DEMAND_LOCATION_ENCODER_PATH = "./models/kavindi/demand/betel_label_encoders.pkl"
DEMAND_LOCATION_PREPROCESSOR_PATH = "./models/kavindi/demand/demand_preprocessor.pkl"

MARKET_DEMAND_MODEL_PATH = "./models/kavindi/market/betel_insights_model_enhanced.pkl" 
MARKET_DEMAND_ENCODER_PATH = "./models/kavindi/market/betel_insights_encoders.pkl"

# Load the price prediction model and encoders
try:
    price_model = joblib.load(PRICE_MODEL_PATH)
    price_encoders = joblib.load(PRICE_ENCODER_PATH)
    price_preprocessor = joblib.load(PRICE_PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading price prediction resources: {str(e)}")

# Load the demand prediction model and encoders
try:
    demand_model = joblib.load(DEMAND_LOCATION_MODEL_PATH)
    demand_encoders = joblib.load(DEMAND_LOCATION_ENCODER_PATH)
    demand_preprocessor = joblib.load(DEMAND_LOCATION_PREPROCESSOR_PATH)
    demand_scaler = demand_preprocessor['scaler']
    numeric_columns = demand_preprocessor['numeric_columns']
except Exception as e:
    raise RuntimeError(f"Error loading demand prediction resources: {str(e)}")

# Load the market demand prediction model and encoders
try:
    market_demand_model = joblib.load(MARKET_DEMAND_MODEL_PATH)
    market_demand_encoders = joblib.load(MARKET_DEMAND_ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading market demand prediction resources: {str(e)}")


# Input schema for price prediction
class PricePredictionInput(BaseModel):
    Date: str  # Format: YYYY-MM-DD
    Leaf_Type: str
    Leaf_Size: str
    Quality_Grade: str
    No_of_Leaves: int
    Location: str
    Season: str


# Input schema for demand prediction
class DemandPredictionInput(BaseModel):
    Date: str
    No_of_Leaves: int
    Leaf_Type: str
    Leaf_Size: str
    Quality_Grade: str


# Function to predict price per leaf
def predict_price(date, leaf_type, leaf_size, quality_grade, no_of_leaves, location, season):
    try:
        # Convert the date to numeric month
        month = pd.to_datetime(date).month

        # Encode categorical features
        encoded_leaf_type = price_encoders['Leaf_Type'].transform([leaf_type])[0]
        encoded_leaf_size = price_encoders['Leaf_Size'].transform([leaf_size])[0]
        encoded_quality_grade = price_encoders['Quality_Grade'].transform([quality_grade])[0]
        encoded_location = price_encoders['Location'].transform([location])[0]
        if 'Season' in price_encoders:
            encoded_season = price_encoders['Season'].transform([season])[0]
        else:
            raise ValueError("Season encoder is missing or not used in the model.")

        # Prepare the feature vector
        features = [[month, encoded_leaf_type, encoded_leaf_size, encoded_quality_grade, no_of_leaves, encoded_location, encoded_season]]

        # Predict the price
        predicted_price = price_model.predict(features)[0]
        rounded_price = round(predicted_price * 2) / 2
        return f"{rounded_price:.2f}"
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'Error in price prediction': str(e)}), 500


# Function to predict highest demand location
def predict_demand_location(date, no_of_leaves, leaf_type, leaf_size, quality_grade):
    try:
        # Convert the date to numeric month
        month = pd.to_datetime(date).month

        # Encode categorical features
        encoded_leaf_type = demand_encoders['Leaf_Type'].transform([leaf_type])[0]
        encoded_leaf_size = demand_encoders['Leaf_Size'].transform([leaf_size])[0]
        encoded_quality_grade = demand_encoders['Quality_Grade'].transform([quality_grade])[0]

        # Prepare the feature vector
        features = pd.DataFrame([[month, no_of_leaves, encoded_leaf_type, encoded_leaf_size, encoded_quality_grade]],
                                columns=['Month', 'No_of_Leaves', 'Leaf_Type', 'Leaf_Size', 'Quality_Grade'])

        # Scale numeric features
        features[numeric_columns] = demand_scaler.transform(features[numeric_columns])

        # Predict location
        location_encoded = demand_model.predict(features)[0]
        location = demand_encoders['Location'].inverse_transform([location_encoded])[0]
        return location
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'Error in demand location': str(e)}), 500

# Function to predict market demand
def predict_market_demand():
    try:
        # Predict market demand
        return 1000
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'Error in market demand prediction': str(e)}), 500

