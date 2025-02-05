from app import app  # Import the app from __init__.py
import os

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))  # Get Railway-assigned port
    app.run(host="0.0.0.0", port=PORT, debug=True)