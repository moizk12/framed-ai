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
    # Get the directory containing this file (framed package)
    from pathlib import Path
    base_dir = Path(__file__).parent
    
    app = Flask(
        __name__,
        template_folder=str(base_dir / 'templates'),
        static_folder=str(base_dir / 'static')
    )
    
    # Basic configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    # Use centralized runtime directory from vision.py
    from framed.analysis.vision import UPLOAD_DIR
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
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
    
    # Health check endpoint - STEP 4.5: NEVER triggers model loading
    # This endpoint returns instantly and does not import or reference any model getters
    @app.route('/health')
    def health():
        """
        Health check for monitoring.
        Returns instantly without loading any models.
        Safe for cold-start verification.
        """
        return {'status': 'healthy', 'service': 'framed'}, 200
    
    return app