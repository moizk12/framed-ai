"""Flask application factory."""
import os
from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """Create and configure the Flask app."""
    from pathlib import Path
    repo_root = Path(__file__).parent.parent

    app = Flask(
        __name__,
        template_folder=str(repo_root / 'templates'),
        static_folder=str(repo_root / 'static'),
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