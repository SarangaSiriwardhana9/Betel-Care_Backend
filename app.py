from app import app
import os

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))  # Railway defaults to 8080
    app.run(host='0.0.0.0', port=port)
