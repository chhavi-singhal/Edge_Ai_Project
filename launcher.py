#!/usr/bin/env python3
"""
Interactive Launcher for Hard Hat Detection System
Select and run different scripts easily
"""

import subprocess
import sys
import os

def print_header():
    """Print the header"""
    print("\n" + "="*60)
    print("   HARD HAT DETECTION SYSTEM - RASPBERRY PI")
    print("   YOLOv8 TFLite Model (25MB)")
    print("="*60 + "\n")

def print_menu():
    """Print the main menu"""
    print("Select an option:")
    print()
    print("  1. Inspect Model (check model details)")
    print("  2. Test Camera (verify camera is working)")
    print("  3. Test Detection on Image (single image detection)")
    print("  4. Live Detection (real-time camera detection)")
    print("  5. Live Detection + Save Video")
    print()
    print("  9. Run Setup Script (install dependencies)")
    print("  0. Exit")
    print()

def check_file_exists(filepath):
    """Check if a file exists"""
    if not os.path.exists(filepath):
        print(f"\n❌ Error: File not found - {filepath}")
        print(f"   Please ensure the file is in the current directory.")
        return False
    return True

def run_script(script_name, args=[]):
    """Run a Python script"""
    if not check_file_exists(script_name):
        return
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    try:
        cmd = ["python3", script_name] + args
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running script: {e}")
    
    print(f"\n{'='*60}\n")
    input("Press Enter to continue...")

def main():
    """Main function"""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("Enter your choice (0-9): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!\n")
                break
            
            elif choice == "1":
                # Inspect Model
                run_script("01_inspect_model.py")
            
            elif choice == "2":
                # Test Camera
                print("\nCamera Test Options:")
                print("  1. Basic test (save test image)")
                print("  2. Live preview test")
                test_choice = input("Choose (1/2): ").strip()
                
                if test_choice == "1":
                    run_script("02_camera_test.py", ["--no-live"])
                else:
                    run_script("02_camera_test.py")
            
            elif choice == "3":
                # Test on Image
                image_path = input("\nEnter image path (or press Enter for default): ").strip()
                if not image_path:
                    image_path = "test_image.jpg"
                
                if os.path.exists(image_path):
                    run_script("03_helmet_detector.py", [image_path])
                else:
                    print(f"\n❌ Image not found: {image_path}")
                    input("Press Enter to continue...")
            
            elif choice == "4":
                # Live Detection
                print("\nStarting live detection...")
                print("Press 'q' to quit, 's' to save snapshot")
                input("Press Enter to start...")
                run_script("04_live_detection.py")
            
            elif choice == "5":
                # Live Detection + Save Video
                print("\nStarting live detection with video recording...")
                print("Press 'q' to quit, 's' to save snapshot")
                print("Video will be saved as 'output.avi'")
                input("Press Enter to start...")
                run_script("04_live_detection.py", ["--save-video"])
            
            elif choice == "9":
                # Run Setup
                if check_file_exists("setup_rpi.sh"):
                    print("\n⚠️  This will install system dependencies.")
                    print("Continue? (y/n): ", end="")
                    confirm = input().strip().lower()
                    
                    if confirm == 'y':
                        try:
                            subprocess.run(["bash", "setup_rpi.sh"])
                        except Exception as e:
                            print(f"\n❌ Error running setup: {e}")
                    
                    input("\nPress Enter to continue...")
            
            else:
                print("\n❌ Invalid choice. Please enter 0-9.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    # Check if we're in the right directory
    required_files = [
        "helmet_detector.py",
        "01_inspect_model.py",
        "02_camera_test.py",
        "03_helmet_detector.py",
        "04_live_detection.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("\n⚠️  Warning: Some required files are missing:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nMake sure you're in the correct directory with all project files.")
        print()
        input("Press Enter to continue anyway...")
    
    # Check for model file
    if not os.path.exists("best_int8.tflite"):
        print("\n⚠️  Warning: Model file 'best_int8.tflite' not found!")
        print("   Please copy your model file to this directory.")
        print()
        input("Press Enter to continue anyway...")
    
    main()
