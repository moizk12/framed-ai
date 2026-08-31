"""Shared pytest hooks for all repository test roots."""

from __future__ import annotations

import faulthandler
import gc
import signal

import pytest

faulthandler.enable()
if hasattr(signal, "SIGUSR1") and hasattr(faulthandler, "register"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)


@pytest.fixture(autouse=True)
def _release_cognition_ledger_singleton():
    """Drop the cognition ledger singleton so SQLite handles do not leak across tests."""
    yield
    try:
        from framed.cognition.ledger.sqlite_store import clear_ledger
    except ImportError:
        return
    clear_ledger()
    gc.collect()
