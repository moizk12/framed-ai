# test_imports.py
import sys
import io

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    from framed import create_app
    print("✓ framed.__init__ import works")
    
    from framed.routes import main
    print("✓ framed.routes import works")
    
    from framed.analysis.vision import analyze_image
    print("✓ framed.analysis.vision import works")
    
    app = create_app()
    print("✓ App creation works")
    
    print("\n[SUCCESS] All imports successful! Ready to deploy.")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")