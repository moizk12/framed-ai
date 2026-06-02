"""Flask application factory."""
import os
from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """Create and configure the Flask app."""
    from pathlib import Path
    base_dir = Path(__file__).parent
    
    app = Flask(
        __name__,
        template_folder=str(base_dir / 'templates'),
        static_folder=str(base_dir / 'static')
    )
    
    # Basic configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    from framed.analysis.vision import UPLOAD_DIR
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    if config:
        app.config.update(config)
    
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    from framed.routes import main
    app.register_blueprint(main)
    
    try:
        from framed.analysis.vision import ensure_directories
        with app.app_context():
            ensure_directories()
    except Exception as e:
        app.logger.warning(f"Could not pre-create directories: {e}")
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return {'status': 'healthy', 'service': 'framed'}, 200
    
    return app