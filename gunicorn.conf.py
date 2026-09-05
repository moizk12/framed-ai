"""One-process synchronous beta budgets, shared with rendered browser config."""
from framed.public_runtime import runtime_defaults

_runtime = runtime_defaults()
timeout = _runtime["PUBLIC_WORKER_TIMEOUT_SECONDS"]
graceful_timeout = timeout
workers = 1
worker_class = "gthread"
threads = 4
keepalive = 5


def post_worker_init(worker):
    from framed.public_runtime import env_bool
    if env_bool("PUBLIC_PRELOAD_MODELS", True):
        try:
            from framed.analysis.models import warm_public_models
            warm_public_models()
        except Exception:
            worker.log.exception("Public model warmup failed; inference readiness remains unavailable")
