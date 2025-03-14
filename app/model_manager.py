import joblib
import tensorflow as tf

# Load XGBoost model (can be extended for other models)
def load_model(model_path):
    if model_path.endswith('.pkl'):
        return joblib.load(model_path)
    elif model_path.endswith('.h5'):
        return tf.keras.models.load_model(model_path)  # TensorFlow model loading

def predict_yield(model, X):
    return model.predict(X)
