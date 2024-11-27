import traceback
from flask import jsonify

# Utility function for error handling
def handle_error(exception):
    # Log the error details (you can expand this for logging)
    error_message = str(exception)
    error_trace = traceback.format_exc()

    return jsonify({
        'error': error_message,
        'trace': error_trace
    }), 500
