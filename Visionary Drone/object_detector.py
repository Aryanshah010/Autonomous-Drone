from ultralytics import YOLO
import cv2
import config
import numpy as np

class ObjectDetector:
    def __init__(self):
        """Initialize YOLOv8 model."""
        try:
            self.model = YOLO(config.MODEL_PATH)
            self.confidence_threshold = config.CONFIDENCE_THRESHOLD
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print("Error loading YOLOv8 model:", str(e))
            self.model = None

    def detect_objects(self, frame):
        """Detect objects in a frame and draw bounding boxes."""
        if self.model is None:
            print("Error: YOLOv8 model not loaded")
            return frame, []
        
        try:
            # Resize frame to reduce processing load
            frame_small = cv2.resize(frame, (640, 360))
            
            # Run YOLOv8 detection
            results = self.model(frame_small, conf=self.confidence_threshold, verbose=False)
            
            # Extract detections
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls = int(box.cls)
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    detections.append({
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                        "confidence": conf,
                        "class_id": cls,
                        "label": label
                    })
                    
                    # Draw bounding box and label
                    if config.DEBUG:
                        cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_small, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Resize back to original size for display
            frame_display = cv2.resize(frame_small, config.RESOLUTION)
            return frame_display, detections
        except Exception as e:
            print("Object detection error:", str(e))
            return frame, []