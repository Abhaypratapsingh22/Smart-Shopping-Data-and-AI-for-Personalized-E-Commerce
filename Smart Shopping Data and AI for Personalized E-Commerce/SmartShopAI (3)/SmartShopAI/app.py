# Main entry point for the Flask application
from app_flask import app, initialize_database
import os

# Make sure database and directories exist
initialize_database()
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)