"""
FRAMED - AI-Powered Photography Analysis Platform
Flask Application Factory

⚠️ CRITICAL FILE - This must exist for the app to work!
"""
import os
from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """
    Application factory pattern for Flask
    
    This creates and configures the Flask application.
    Called by run.py when starting the server.
    """
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static'
    )
    
    # Basic configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_DIR', '/data/uploads')
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Apply any additional config
    if config:
        app.config.update(config)
    
    # Enable CORS for all routes
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Register blueprints (routes)
    from framed.routes import main
    app.register_blueprint(main)
    
    # Ensure necessary directories exist
    try:
        from framed.analysis.vision import ensure_directories
        with app.app_context():
            ensure_directories()
    except Exception as e:
        app.logger.warning(f"Could not pre-create directories: {e}")
    
    # Health check endpoint
    @app.route('/health')
    def health():
        """Health check for monitoring"""
        return {'status': 'healthy', 'service': 'framed'}, 200
    
    return app