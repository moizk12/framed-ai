# test_imports.py
try:
    from framed import create_app
    print("✅ framed.__init__ import works")
    
    from framed.routes import main
    print("✅ framed.routes import works")
    
    from framed.analysis.vision import analyze_image
    print("✅ framed.analysis.vision import works")
    
    app = create_app()
    print("✅ App creation works")
    
    print("\n🎉 All imports successful! Ready to deploy.")
except ImportError as e:
    print(f"❌ Import error: {e}")