import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, force CPU for TensorFlow

from flask import Flask
from app.routes import api_bp
from app.watering_routes import watering_bp
from app.fertilizing_routes import fertilizing_bp
from app.protection_routes import protection_bp  # Import the new Protection Blueprint

app = Flask(__name__)

# Register the original API Blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Register the Watering Blueprint
app.register_blueprint(watering_bp, url_prefix='/api/watering')

# Register the Fertilizing Blueprint
app.register_blueprint(fertilizing_bp, url_prefix='/api/fertilizing')

# Register the Protection Blueprint
app.register_blueprint(protection_bp, url_prefix='/api/protection')

# Create necessary directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)