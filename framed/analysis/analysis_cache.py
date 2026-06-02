# REF:D2 SHA256-keyed analysis JSON cache (schema-validated)
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

from .schema import validate_schema
from .runtime_paths import ANALYSIS_CACHE_DIR, CACHE_VERSION, ensure_directories

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """SHA-256 of file contents; empty string on error."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute file hash: {e}", exc_info=True)
        return ""


def get_cached_analysis(file_hash: str) -> Optional[Dict[str, Any]]:
    if not file_hash:
        return None
    cache_path = os.path.join(ANALYSIS_CACHE_DIR, f"{file_hash}.json")
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            cached_result = json.load(f)
        if cached_result.get("_cache_version") != CACHE_VERSION:
            logger.info(f"Cache invalidated for {file_hash[:8]}... (version mismatch)")
            return None
        cached_result.pop("_cache_version", None)
        if validate_schema(cached_result):
            logger.info(f"Cache hit for hash: {file_hash[:8]}...")
            return cached_result
        logger.warning(f"Cached result for {file_hash[:8]}... failed schema validation, ignoring")
        return None
    except Exception as e:
        logger.error(f"Failed to read cache for {file_hash[:8]}...: {e}", exc_info=True)
        return None


def save_cached_analysis(file_hash: str, result: Dict[str, Any]) -> bool:
    if not file_hash:
        return False
    cache_path = os.path.join(ANALYSIS_CACHE_DIR, f"{file_hash}.json")
    try:
        ensure_directories()
        to_save = dict(result)
        to_save["_cache_version"] = CACHE_VERSION
        with open(cache_path, "w") as f:
            json.dump(to_save, f, indent=2, default=str)
        logger.info(f"Cached analysis for hash: {file_hash[:8]}...")
        return True
    except Exception as e:
        logger.error(f"Failed to save cache for {file_hash[:8]}...: {e}", exc_info=True)
        return False
