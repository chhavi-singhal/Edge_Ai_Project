"""
Test installation of all dependencies
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("\n" + "="*50)
    print("Testing Package Imports")
    print("="*50 + "\n")
    
    packages = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('tensorflow.lite', 'TensorFlow Lite')
    ]
    
    success = True
    
    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED: {e}")
            success = False
    
    print("\n" + "="*50)
    
    if success:
        print("\n✓ All packages imported successfully!")
        print("\nYou can now run the detection scripts.")
    else:
        print("\n✗ Some packages failed to import.")
        print("Please install missing packages manually.")
    
    print("="*50 + "\n")
    
    return success

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
