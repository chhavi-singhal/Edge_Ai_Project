#!/bin/bash

###############################################################################
# Raspberry Pi Hard Hat Detection Setup Script
# This script installs all dependencies for running the 25MB TFLite model
###############################################################################

echo "=========================================="
echo "Hard Hat Detection - Raspberry Pi Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check if running on Raspberry Pi
print_info "Checking system..."
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    print_success "Detected: $MODEL"
else
    print_info "Note: Not running on Raspberry Pi, but continuing anyway..."
fi

# Update package lists
print_info "Updating package lists..."
sudo apt-get update
print_success "Package lists updated"

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

print_success "System dependencies installed"

# Upgrade pip
print_info "Upgrading pip..."
python3 -m pip install --upgrade pip
print_success "pip upgraded"

# Install Python packages
print_info "Installing Python packages..."
pip3 install --upgrade \
    numpy \
    opencv-python \
    opencv-contrib-python \
    pillow \
    tflite-runtime

# If tflite-runtime fails, try tensorflow-lite
if [ $? -ne 0 ]; then
    print_info "Installing tensorflow-lite as fallback..."
    pip3 install --upgrade tensorflow-lite
fi

print_success "Python packages installed"

# Create directories
print_info "Creating project directories..."
mkdir -p ~/hardhat-detection
mkdir -p ~/hardhat-detection/captures
mkdir -p ~/hardhat-detection/videos
mkdir -p ~/hardhat-detection/logs

print_success "Directories created"

# Enable camera (for Raspberry Pi Camera Module)
print_info "Checking camera configuration..."
if command -v raspi-config &> /dev/null; then
    print_info "Raspberry Pi detected - enabling camera..."
    sudo raspi-config nonint do_camera 0
    print_success "Camera enabled (reboot may be required)"
else
    print_info "Not a Raspberry Pi - skipping camera config"
fi

# Create a test script to verify installation
print_info "Creating test script..."
cat > ~/hardhat-detection/test_installation.py << 'EOF'
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
EOF

print_success "Test script created"

# Run the test
print_info "Testing installation..."
python3 ~/hardhat-detection/test_installation.py

# Print completion message
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
print_success "All dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "  1. Copy all Python scripts to ~/hardhat-detection/"
echo "  2. Copy best_int8.tflite to ~/hardhat-detection/"
echo "  3. Test camera: python3 02_camera_test.py"
echo "  4. Run detection: python3 04_live_detection.py"
echo ""
echo "For more info, see README.md"
echo ""

# Check if reboot is needed
if [ -f /var/run/reboot-required ]; then
    print_info "System reboot recommended"
    echo ""
    echo "Reboot now? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        sudo reboot
    fi
fi
