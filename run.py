import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask app is running!"

PORT = int(os.environ.get("PORT", 5000))  # Get Railway-assigned port
app.run(host="0.0.0.0", port=PORT, debug=True)

