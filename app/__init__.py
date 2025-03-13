from flask import Flask
from app.routes import api_bp

app = Flask(__name__)

# Register the API Blueprint
app.register_blueprint(api_bp, url_prefix='/api')