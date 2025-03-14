import joblib
import tensorflow as tf
import xgboost as xgb
from typing import Any, Union

def load_model(model_path: str) -> Any:
    """Load an ML model based on the file extension."""
    try:
        if model_path.endswith(".pkl"):
            return joblib.load(model_path, compile=False)  # Avoid GPU-related issue
        elif model_path.endswith(".h5"):
            return tf.keras.models.load_model(model_path, compile=False)  # Avoid GPU-related issues
        elif model_path.endswith(".json"):
            with open(model_path, "r") as f:
                model_json = f.read()
            model = tf.keras.models.model_from_json(model_json)
            return model
        else:
            raise ValueError("Unsupported model format. Use .pkl, .h5, or .json.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_yield(model: Any, X: Any) -> Union[Any, None]:
    """Make predictions using the loaded model."""
    try:
        return model.predict(X)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
