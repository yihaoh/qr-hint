# path: backend/app.py
from flask import Flask
from flask_cors import CORS
from config import get_config
import os

def create_app(config_name=None):
    """Application factory pattern"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    app = Flask(__name__)

    # Load configuration
    config_obj = get_config(config_name)
    app.config.from_object(config_obj)

    # Enable CORS
    CORS(app, origins=config_obj.CORS_ORIGINS)

    # Register blueprints
    from api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # Health check route
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'message': 'QR-Hint backend is running'}, 200
    
    

    return app

if __name__ == '__main__':
    app = create_app()
    config_obj = get_config(os.getenv('FLASK_ENV', 'development'))
    app.run(
        host=config_obj.HOST,
        port=config_obj.PORT,
        debug=config_obj.DEBUG
    )
