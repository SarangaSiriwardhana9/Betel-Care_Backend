from flask import Flask
from flask_cors import CORS
from app.routes import api_bp

app = Flask(__name__)

# Enable CORS to allow requests from other devices
CORS(app)

# Register the API Blueprint
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port , debug=True)