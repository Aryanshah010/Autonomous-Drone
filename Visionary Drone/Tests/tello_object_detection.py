from drone_controller import DroneController
from video_processor import VideoProcessor
from object_detector import ObjectDetector
import time
import threading
import sys
import config
import cv2

def main():
    # Initialize drone controller
    controller = DroneController()
    
    # Connect to drone
    if not controller.connect():
        print("Exiting due to connection failure")
        sys.exit(1)
    
    # Get initial battery level
    battery = controller.get_battery()
    if battery is None or battery < 20:
        print("Battery too low or unavailable, exiting")
        controller.disconnect()
        sys.exit(1)
    
    # Start status monitoring in a thread
    status_thread = threading.Thread(target=controller.monitor_status, args=(5,))
    status_thread.daemon = True
    status_thread.start()
    
    # Initialize video processor and object detector
    video_processor = VideoProcessor(controller)
    detector = ObjectDetector()
    
    # Start video stream
    time.sleep(2)  # Delay for connection stability
    if not video_processor.start_stream():
        print("Video stream failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    # Perform takeoff
    time.sleep(2)  # Delay for stream stability
    if not controller.takeoff():
        print("Takeoff failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    # Process video with object detection for 5 seconds
    print("Detecting objects for 5 seconds. Press 'q' to stop early.")
    start_time = time.time()
    try:
        while time.time() - start_time < 5:
            if not video_processor.display_frame():
                print("Video display failed, landing drone")
                break
            ret, frame = video_processor.video_capture.read()
            if not ret:
                print("Warning: Could not read frame, skipping")
                continue
            # Detect objects
            annotated_frame, detections = detector.detect_objects(frame)
            if config.DEBUG:
                cv2.imshow("Tello Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Print detections for debugging
            if detections:
                print("Detected:", [d["label"] for d in detections])
            time.sleep(0.1)  # Reduce CPU load
    except KeyboardInterrupt:
        print("User interrupted, landing drone")
    
    # Land the drone
    time.sleep(1)  # Delay to ensure video stops
    if not controller.land():
        print("Landing failed, attempting emergency stop")
        controller.emergency_stop()
    
    # Stop video stream
    video_processor.stop_stream()
    
    # Get final battery level
    controller.get_battery()
    
    # Disconnect
    controller.disconnect()

if __name__ == "__main__":
    main()