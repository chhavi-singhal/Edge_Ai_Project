"""
Hard Hat Detector Module
Standalone detector class that can be imported by other scripts
"""

import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time

class HardHatDetector:
    """
    YOLOv8 TFLite Hard Hat Detector
    Detects: Hardhat, NO-Hardhat
    """
    
    def __init__(self, model_path="best_int8.tflite", conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the Hard Hat Detector
        
        Args:
            model_path: Path to the TFLite model
            conf_threshold: Confidence threshold for detections (default: 0.5)
            iou_threshold: IoU threshold for NMS (default: 0.45)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names (must match training data)
        self.class_names = {
            0: "Hardhat",
            1: "NO-Hardhat"
        }
        
        # Colors for visualization (BGR format)
        self.colors = {
            0: (0, 255, 0),    # Green for Hardhat (safe)
            1: (0, 0, 255)     # Red for NO-Hardhat (unsafe)
        }
        
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
        
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: Input image in BGR format (from OpenCV)
            
        Returns:
            input_data: Preprocessed image ready for inference
            scale_x: Width scaling factor
            scale_y: Height scaling factor
        """
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Resize to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension [1, height, width, channels]
        input_data = np.expand_dims(normalized, axis=0)
        
        # Calculate scaling factors to map predictions back to original image
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        return input_data, scale_x, scale_y
    
    def postprocess_predictions(self, outputs, scale_x, scale_y, orig_width, orig_height):
        """
        Process model outputs to extract bounding boxes and classes
        
        Args:
            outputs: Raw model outputs
            scale_x: Width scaling factor
            scale_y: Height scaling factor
            orig_width: Original image width
            orig_height: Original image height
            
        Returns:
            detections: List of [class_id, confidence, x1, y1, x2, y2]
        """
        detections = []
        
        # YOLOv8 TFLite output format: [1, 6, 8400]
        # 6 channels: [x_center, y_center, width, height, conf_class0, conf_class1]
        # 8400 predictions from different anchor points
        output_data = outputs[0]
        
        # Transpose to [8400, 6] for easier processing
        output_data = output_data[0].T
        
        for detection in output_data:
            # Parse detection
            x_center, y_center, width, height, conf_hardhat, conf_no_hardhat = detection
            
            # Determine class with highest confidence
            confidences = [conf_hardhat, conf_no_hardhat]
            class_id = np.argmax(confidences)
            confidence = confidences[class_id]
            
            # Filter by confidence threshold
            if confidence < self.conf_threshold:
                continue
            
            # Convert from center format (x_center, y_center, w, h) 
            # to corner format (x1, y1, x2, y2)
            x1 = (x_center - width / 2) * scale_x
            y1 = (y_center - height / 2) * scale_y
            x2 = (x_center + width / 2) * scale_x
            y2 = (y_center + height / 2) * scale_y
            
            # Clip coordinates to image boundaries
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))
            
            detections.append([class_id, confidence, x1, y1, x2, y2])
        
        # Apply Non-Maximum Suppression to remove overlapping boxes
        if len(detections) > 0:
            detections = self.non_max_suppression(detections)
        
        return detections
    
    def non_max_suppression(self, detections):
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detections
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        detections = np.array(detections)
        
        # Extract boxes and scores
        boxes = detections[:, 2:6].astype(float)  # [x1, y1, x2, y2]
        scores = detections[:, 1].astype(float)   # confidence scores
        
        # Apply OpenCV's NMS
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
            image: Input image in BGR format (from OpenCV)
            
        Returns:
            detections: List of detections [class_id, confidence, x1, y1, x2, y2]
            inference_time: Time taken for inference (seconds)
        """
        start_time = time.time()
        
        # Preprocess image
        input_data, scale_x, scale_y = self.preprocess_image(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output)
        
        # Postprocess predictions
        orig_height, orig_width = image.shape[:2]
        detections = self.postprocess_predictions(
            outputs, scale_x, scale_y, orig_width, orig_height
        )
        
        inference_time = time.time() - start_time
        
        return detections, inference_time
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image
        
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
            
            # Get class information
            class_name = self.class_names[class_id]
            color = self.colors[class_id]
            
            # Draw bounding box
            cv2.rectangle(
                result,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get label size for background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                2
            )
            
            # Draw label background
            cv2.rectangle(
                result,
                (int(x1), int(y1) - label_height - 10),
                (int(x1) + label_width, int(y1)),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                result,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
        
        return result
    
    def get_detection_summary(self, detections):
        """
        Get a summary of detections
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with detection counts
        """
        summary = {
            "total": len(detections),
            "Hardhat": 0,
            "NO-Hardhat": 0,
            "detections": []
        }
        
        for detection in detections:
            class_id, confidence, x1, y1, x2, y2 = detection
            class_name = self.class_names[int(class_id)]
            
            summary[class_name] += 1
            summary["detections"].append({
                "class": class_name,
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        
        return summary
