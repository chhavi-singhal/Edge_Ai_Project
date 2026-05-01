"""
Hard Hat Detection using YOLOv8 TFLite Model (25MB version)
Detects Hardhat and NO-Hardhat on images
"""

import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from datetime import datetime
import time

class HardHatDetector:
    def __init__(self, model_path="best_int8.tflite", conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the Hard Hat Detector
        
        Args:
            model_path: Path to the TFLite model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names for the model
        self.class_names = {
            0: "Hardhat",
            1: "NO-Hardhat"
        }
        
        # Colors for bounding boxes (BGR format)
        self.colors = {
            0: (0, 255, 0),    # Green for Hardhat
            1: (0, 0, 255)     # Red for NO-Hardhat
        }
        
        print(f"🔧 Initializing Hard Hat Detector...")
        print(f"   Model: {model_path}")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   IoU threshold: {iou_threshold}")
        
        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Expected input size: {self.input_width}x{self.input_height}")
        print(f"   Output tensors: {len(self.output_details)}")
        
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image, scale factors
        """
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        # Calculate scale factors
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        return input_data, scale_x, scale_y
    
    def postprocess_predictions(self, outputs, scale_x, scale_y, orig_width, orig_height):
        """
        Process model outputs to get bounding boxes
        
        Args:
            outputs: Model output tensors
            scale_x: Width scale factor
            scale_y: Height scale factor
            orig_width: Original image width
            orig_height: Original image height
            
        Returns:
            List of detections [class_id, confidence, x1, y1, x2, y2]
        """
        detections = []
        
        # YOLOv8 TFLite output format: [1, 6, 8400]
        # where 6 = [x, y, w, h, conf_class0, conf_class1]
        output_data = outputs[0]  # Shape: [1, 6, 8400]
        
        # Transpose to [8400, 6]
        output_data = output_data[0].T  # Now shape: [8400, 6]
        
        for detection in output_data:
            x_center, y_center, width, height, conf_hardhat, conf_no_hardhat = detection
            
            # Get class with highest confidence
            confidences = [conf_hardhat, conf_no_hardhat]
            class_id = np.argmax(confidences)
            confidence = confidences[class_id]
            
            # Filter by confidence threshold
            if confidence < self.conf_threshold:
                continue
            
            # Convert from center format to corner format
            x1 = (x_center - width / 2) * scale_x
            y1 = (y_center - height / 2) * scale_y
            x2 = (x_center + width / 2) * scale_x
            y2 = (y_center + height / 2) * scale_y
            
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))
            
            detections.append([class_id, confidence, x1, y1, x2, y2])
        
        # Apply Non-Maximum Suppression
        if len(detections) > 0:
            detections = self.non_max_suppression(detections)
        
        return detections
    
    def non_max_suppression(self, detections):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return []
        
        # Convert to numpy array
        detections = np.array(detections)
        
        # Extract boxes and scores
        boxes = detections[:, 2:6]  # [x1, y1, x2, y2]
        scores = detections[:, 1]   # confidence
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) > 0:
            return detections[indices.flatten()].tolist()
        else:
            return []
    
    def detect(self, image):
        """
        Run detection on an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            detections: List of detections
            inference_time: Inference time in seconds
        """
        # Start timing
        start_time = time.time()
        
        # Preprocess
        input_data, scale_x, scale_y = self.preprocess_image(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output)
        
        # Postprocess
        orig_height, orig_width = image.shape[:2]
        detections = self.postprocess_predictions(
            outputs, scale_x, scale_y, orig_width, orig_height
        )
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        return detections, inference_time
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for detection in detections:
            class_id, confidence, x1, y1, x2, y2 = detection
            class_id = int(class_id)
            
            # Get class name and color
            class_name = self.class_names[class_id]
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(result, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(result,
                         (int(x1), int(y1) - label_height - 10),
                         (int(x1) + label_width, int(y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(result, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
        
        return result

def test_on_image(image_path, model_path="best_int8.tflite"):
    """
    Test detector on a single image
    """
    print(f"\n{'='*60}")
    print(f"TESTING ON IMAGE: {image_path}")
    print(f"{'='*60}\n")
    
    # Initialize detector
    detector = HardHatDetector(model_path)
    
    # Load image
    print(f"📷 Loading image...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Error: Could not load image '{image_path}'")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Run detection
    print(f"\n🔍 Running detection...")
    detections, inference_time = detector.detect(image)
    
    print(f"✅ Detection complete!")
    print(f"   Inference time: {inference_time*1000:.2f} ms")
    print(f"   Detections found: {len(detections)}")
    
    # Print detections
    if len(detections) > 0:
        print(f"\n📋 Detection Results:")
        for i, detection in enumerate(detections):
            class_id, confidence, x1, y1, x2, y2 = detection
            class_name = detector.class_names[int(class_id)]
            print(f"   {i+1}. {class_name} (conf: {confidence:.3f}) at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    else:
        print(f"\n⚠️  No detections found")
    
    # Draw detections
    result = detector.draw_detections(image, detections)
    
    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"detection_result_{timestamp}.jpg"
    cv2.imwrite(output_path, result)
    print(f"\n💾 Result saved as: {output_path}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    # Check if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use test image if available
        image_path = "test_image.jpg"
        print(f"No image path provided. Using default: {image_path}")
        print(f"Usage: python 03_helmet_detector.py <image_path>")
    
    # Run test
    test_on_image(image_path)
