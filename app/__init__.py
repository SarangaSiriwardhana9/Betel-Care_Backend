import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   
import tensorflow as tf
from flask import Flask
from app.routes import api_bp
from app.watering_routes import watering_bp
from app.fertilizing_routes import fertilizing_bp
from app.protection_routes import protection_bp

app = Flask(__name__)

# Register the API Blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Register your additional Blueprints
app.register_blueprint(watering_bp, url_prefix='/api/watering')
app.register_blueprint(fertilizing_bp, url_prefix='/api/fertilizing')
app.register_blueprint(protection_bp, url_prefix='/api/protection')

# Create necessary directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)