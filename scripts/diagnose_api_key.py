#!/usr/bin/env python3
"""
Diagnose OPENAI_API_KEY loading issues.

Run from project root: python scripts/diagnose_api_key.py
"""

import os
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    
    print("=" * 60)
    print("OPENAI_API_KEY Diagnostic")
    print("=" * 60)
    print(f"Project root: {root}")
    print(f"CWD:         {os.getcwd()}")
    print()
    
    # 1. Check .env exists
    env_path = root / ".env"
    print(f"1. .env file exists: {env_path.exists()}")
    if not env_path.exists():
        print("   -> Create .env from .env.example and add your key.")
        return 1
    print(f"   Path: {env_path}")
    print()
    
    # 2. Check .env contents (without revealing key)
    try:
        raw = env_path.read_text(encoding="utf-8-sig")  # utf-8-sig strips BOM
        lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("#")]
        has_key = any("OPENAI_API_KEY" in l for l in lines)
        key_line = next((l for l in lines if "OPENAI_API_KEY" in l and "=" in l), None)
        print(f"2. OPENAI_API_KEY line in .env: {bool(key_line)}")
        if key_line:
            parts = key_line.split("=", 1)
            val = parts[1].strip().strip('"').strip("'").strip() if len(parts) > 1 else ""
            masked = f"{val[:7]}...{val[-4:]}" if len(val) > 15 else "(too short)"
            print(f"   Value length: {len(val)} chars")
            print(f"   Masked: {masked}")
            if len(val) < 10:
                print("   -> Key seems too short. Check for typos or truncation.")
            if "your_key_here" in val or "sk-" not in val:
                print("   -> Key looks like placeholder. Replace with real key.")
        else:
            print("   -> Add line: OPENAI_API_KEY=sk-your-actual-key")
        print()
    except Exception as e:
        print(f"   Error reading .env: {e}")
        print()
    
    # 3. Load dotenv
    print("3. Loading with python-dotenv...")
    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(env_path)
        print(f"   load_dotenv returned: {loaded}")
    except ImportError as e:
        print(f"   ERROR: python-dotenv not installed. Run: pip install python-dotenv")
        return 1
    
    # 4. Check os.environ after load
    key = os.getenv("OPENAI_API_KEY", "")
    key_clean = (key or "").strip()
    print(f"4. os.getenv('OPENAI_API_KEY') after load: {'set' if key_clean else 'EMPTY'}")
    if key_clean:
        print(f"   Length: {len(key_clean)} chars")
    else:
        print("   -> Key not in environment. Possible causes:")
        print("      - .env has wrong format (use OPENAI_API_KEY=sk-xxx, no spaces around =)")
        print("      - .env was saved with wrong encoding (use UTF-8)")
        print("      - Another process cleared env (try running in fresh terminal)")
    print()
    
    # 5. Check OpenAI client
    if key_clean:
        print("5. Testing OpenAI client...")
        try:
            from openai import OpenAI
            client = OpenAI()
            # Minimal test - list models or similar (cheap)
            models = list(client.models.list())
            print("   OK: OpenAI client initialized and can list models")
        except Exception as e:
            print(f"   Client error: {e}")
            if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                print("   -> API key may be invalid or revoked.")
            elif "model" in str(e).lower():
                print("   -> Key works but model may not exist (gpt-5.2, gpt-5-mini).")
    
    print("=" * 60)
    return 0 if key_clean else 1


if __name__ == "__main__":
    sys.exit(main())
