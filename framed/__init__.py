"""Flask application factory."""
import os
import tempfile
from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """Create and configure the Flask app."""
    from pathlib import Path
    base_dir = Path(__file__).parent
    
    app = Flask(
        __name__,
        template_folder=str(base_dir / 'templates'),
        static_folder=None,
    )
    
    # Basic configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', '')
    app.config['PUBLIC_AUTO_MIGRATE'] = True
    default_data_dir = os.environ.get("FRAMED_DATA_DIR", os.path.join(tempfile.gettempdir(), "framed"))
    app.config['UPLOAD_FOLDER'] = os.path.join(default_data_dir, "uploads")
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    if config:
        app.config.update(config)
    
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    from framed.routes import main
    app.register_blueprint(main)

    from framed.public_store import PublicBetaStore, build_public_repository
    app.extensions["framed_public_store"] = PublicBetaStore(build_public_repository(app.config))

    from werkzeug.exceptions import RequestEntityTooLarge

    @app.errorhandler(RequestEntityTooLarge)
    def public_payload_too_large(_error):
        from flask import jsonify, request
        from framed.routes import public_error_payload

        if request.path.startswith("/api/v1/"):
            return jsonify(public_error_payload("payload_too_large", "The upload exceeds the allowed size.")), 413
        return {"error": "Payload too large"}, 413
    
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError as e:
        app.logger.warning(f"Could not pre-create upload directory: {e}")
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return {'status': 'healthy', 'service': 'framed'}, 200
    
    return app
