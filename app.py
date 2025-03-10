from app import app  # Import from __init__.py
import os

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Use Railway-assigned PORT or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
