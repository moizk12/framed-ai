"""Flask application factory."""
import os
import tempfile
from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """Create and configure the Flask app."""
    from pathlib import Path
    repo_root = Path(__file__).parent.parent

    app = Flask(
        __name__,
        template_folder=str(repo_root / 'templates'),
        static_folder=None,
    )
    
    from framed.public_runtime import runtime_defaults, validate_runtime

    # Public runtime configuration is environment-driven and validated after
    # callers have had a chance to inject test-only dependencies.
    app.config.update(runtime_defaults())
    app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', '')
    default_data_dir = os.environ.get("FRAMED_DATA_DIR", os.path.join(tempfile.gettempdir(), "framed"))
    app.config['UPLOAD_FOLDER'] = os.path.join(default_data_dir, "uploads")
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    public_boundary_explicit = bool(config and "PUBLIC_BETA_ONLY" in config)
    if config:
        app.config.update(config)
    if app.config.get("TESTING") and not public_boundary_explicit:
        app.config["PUBLIC_BETA_ONLY"] = False

    validate_runtime(app.config)
    from framed.public_limits import AnalysisLimiter
    app.extensions["framed_analysis_limiter"] = AnalysisLimiter(
        app.config["PUBLIC_RATE_LIMIT"], app.config["PUBLIC_RATE_WINDOW_SECONDS"]
    )
    
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

    @app.route('/ready')
    def ready():
        """Readiness includes the public PostgreSQL authority."""
        try:
            app.extensions["framed_public_store"].ready()
            if app.config["FRAMED_ENV"] == "production" and not app.config.get("TESTING"):
                from framed.analysis.models import public_models_ready
                from framed.analysis.llm_provider import get_model_a_provider, get_model_b_provider, PlaceholderProvider
                providers = (get_model_a_provider(), get_model_b_provider())
                if not public_models_ready() or any(isinstance(p, PlaceholderProvider) or not p.is_available() for p in providers):
                    return {'status': 'not_ready', 'service': 'framed'}, 503
        except Exception:
            app.logger.warning("Public persistence readiness check failed", exc_info=True)
            return {'status': 'not_ready', 'service': 'framed'}, 503
        return {'status': 'ready', 'service': 'framed'}, 200

    @app.route('/version')
    def version():
        from framed.public_runtime import safe_version_payload

        return safe_version_payload(app.config), 200
    
    return app
