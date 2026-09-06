"""Fail-closed production configuration for the Track A public beta."""

from __future__ import annotations

import os
import re


DEFAULT_MAX_UPLOAD_BYTES = 12 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 300
DEFAULT_WORKER_TIMEOUT_SECONDS = 360
_MIN_UPLOAD_BYTES = 1024 * 1024
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_SAFE_SHA = re.compile(r"^[0-9a-f]{7,64}$")


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def runtime_defaults() -> dict:
    return {
        "FRAMED_ENV": os.environ.get("FRAMED_ENV", "development").strip().lower(),
        "PUBLIC_ANALYSIS_TIMEOUT_SECONDS": env_int("PUBLIC_ANALYSIS_TIMEOUT_SECONDS", DEFAULT_ANALYSIS_TIMEOUT_SECONDS),
        "PUBLIC_WORKER_TIMEOUT_SECONDS": env_int("PUBLIC_WORKER_TIMEOUT_SECONDS", DEFAULT_WORKER_TIMEOUT_SECONDS),
        "PUBLIC_RATE_LIMIT": env_int("PUBLIC_RATE_LIMIT", 6),
        "PUBLIC_RATE_WINDOW_SECONDS": env_int("PUBLIC_RATE_WINDOW_SECONDS", 600),
        "PUBLIC_BETA_ONLY": env_bool("FRAMED_PUBLIC_BETA_ONLY", True),
        "PUBLIC_AUTO_MIGRATE": env_bool("PUBLIC_AUTO_MIGRATE", False),
        "MAX_CONTENT_LENGTH": env_int("FRAMED_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
        "PUBLIC_MAX_IMAGE_PIXELS": env_int("FRAMED_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS),
        "FRAMED_VERSION": os.environ.get("FRAMED_VERSION", "dev").strip(),
        "FRAMED_BUILD_SHA": os.environ.get("FRAMED_BUILD_SHA", "unknown").strip().lower(),
    }


def validate_runtime(config: dict) -> None:
    for key in ("PUBLIC_ANALYSIS_TIMEOUT_SECONDS", "PUBLIC_WORKER_TIMEOUT_SECONDS", "PUBLIC_RATE_LIMIT", "PUBLIC_RATE_WINDOW_SECONDS"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise RuntimeError(f"{key} must be a positive integer")
    if config["PUBLIC_WORKER_TIMEOUT_SECONDS"] < config["PUBLIC_ANALYSIS_TIMEOUT_SECONDS"] + 30:
        raise RuntimeError("Worker timeout must exceed browser timeout by at least 30 seconds")
    environment = config.get("FRAMED_ENV")
    if environment not in {"development", "test", "production"}:
        raise RuntimeError("FRAMED_ENV must be development, test, or production")

    upload_bytes = config.get("MAX_CONTENT_LENGTH")
    if not isinstance(upload_bytes, int) or upload_bytes <= 0:
        raise RuntimeError("FRAMED_MAX_UPLOAD_BYTES must be a positive integer")
    if not config.get("TESTING") and not _MIN_UPLOAD_BYTES <= upload_bytes <= _MAX_UPLOAD_BYTES:
        raise RuntimeError("FRAMED_MAX_UPLOAD_BYTES must be between 1 MiB and 25 MiB")
    max_pixels = config.get("PUBLIC_MAX_IMAGE_PIXELS")
    if not isinstance(max_pixels, int) or max_pixels <= 0:
        raise RuntimeError("FRAMED_MAX_IMAGE_PIXELS must be a positive integer")
    if not config.get("TESTING") and not 1_000_000 <= max_pixels <= 80_000_000:
        raise RuntimeError("FRAMED_MAX_IMAGE_PIXELS must be between 1,000,000 and 80,000,000")

    version = str(config.get("FRAMED_VERSION") or "")
    build_sha = str(config.get("FRAMED_BUILD_SHA") or "")
    if not _SAFE_VERSION.fullmatch(version):
        raise RuntimeError("FRAMED_VERSION contains unsafe characters")
    if build_sha != "unknown" and not _SAFE_SHA.fullmatch(build_sha):
        raise RuntimeError("FRAMED_BUILD_SHA must be a lowercase Git SHA")

    if environment != "production" or config.get("TESTING"):
        return
    database_url = str(config.get("DATABASE_URL") or "")
    if not database_url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("production DATABASE_URL must use PostgreSQL")
    secret_key = str(config.get("SECRET_KEY") or "")
    if len(secret_key) < 32 or secret_key == "dev-secret-key-change-in-production":
        raise RuntimeError("production SECRET_KEY must be at least 32 characters")
    if not config.get("PUBLIC_BETA_ONLY"):
        raise RuntimeError("production requires FRAMED_PUBLIC_BETA_ONLY=true")
    if env_bool("FRAMED_COGNITION_V1", False):
        raise RuntimeError("FRAMED_COGNITION_V1 must be false in the public production runtime")


def safe_version_payload(config: dict) -> dict:
    return {
        "service": "framed-public-beta",
        "version": config["FRAMED_VERSION"],
        "build_sha": config["FRAMED_BUILD_SHA"],
        "api_contract": "v1",
    }
