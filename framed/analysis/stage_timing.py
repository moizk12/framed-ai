# Optional request-stage timings (FRAMED_LOG_STAGE_TIMINGS=true). See SPEED_AND_EFFICIENCY_ACTION_STEPS.md.
import logging
import os
import time

logger = logging.getLogger(__name__)


def log_stage_done(label: str, t_request_start: float, t_stage_start: float) -> None:
    """Log stage duration and total elapsed since request start (milliseconds)."""
    if os.environ.get("FRAMED_LOG_STAGE_TIMINGS", "").lower() not in ("1", "true", "yes"):
        return
    now = time.perf_counter()
    logger.info(
        "stage_timing label=%s stage_ms=%.1f total_ms=%.1f",
        label,
        (now - t_stage_start) * 1000,
        (now - t_request_start) * 1000,
    )
