#!/usr/bin/env python3
"""Quick check that OPENAI_API_KEY is loaded from .env"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
except ImportError:
    pass
key = os.getenv("OPENAI_API_KEY", "").strip()
if key:
    print("OK: OPENAI_API_KEY is set (%d chars)" % len(key))
else:
    print("FAIL: OPENAI_API_KEY is empty or missing. Check .env in project root.")
