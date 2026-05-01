"""
Inspect TFLite Model - YOLOv8 PPE Detection (25MB version)
This script inspects the model structure and prints detailed information.
"""

#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np

def inspect_tflite_model(model_path):
    """
    Inspect TFLite model and print detailed information
    """
    print(f"\n{'='*60}")
    print(f"INSPECTING MODEL: {model_path}")
    print(f"{'='*60}\n")
    
    # Load TFLite mo
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("📥 INPUT DETAILS:")
    print("-" * 60)
    for i, detail in enumerate(input_details):
        print(f"  Input {i}:")
        print(f"    Name: {detail['name']}")
        print(f"    Shape: {detail['shape']}")
        print(f"    Type: {detail['dtype']}")
        print(f"    Quantization: {detail['quantization']}")
        print()
    
    print("📤 OUTPUT DETAILS:")
    print("-" * 60)
    for i, detail in enumerate(output_details):
        print(f"  Output {i}:")
        print(f"    Name: {detail['name']}")
        print(f"    Shape: {detail['shape']}")
        print(f"    Type: {detail['dtype']}")
        print(f"    Quantization: {detail['quantization']}")
        print()
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    print(f"📊 MODEL STATISTICS:")
    print("-" * 60)
    print(f"  Total tensors: {len(tensor_details)}")
    print(f"  Input tensors: {len(input_details)}")
    print(f"  Output tensors: {len(output_details)}")
    
    # Try to get model size
    import os
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Model size: {size_mb:.2f} MB")
    
    print(f"\n{'='*60}\n")
    
    return interpreter, input_details, output_details

if __name__ == "__main__":
    # Path to your 25MB TFLite model
    MODEL_PATH = "best_int8.tflite"
    
    try:
        interpreter, input_details, output_details = inspect_tflite_model(MODEL_PATH)
        print("✅ Model inspection completed successfully!")
        print("\nExpected classes:")
        print("  Class 0: Hardhat")
        print("  Class 1: NO-Hardhat")
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        print("\nMake sure 'best_int8.tflite' is in the same directory!")
