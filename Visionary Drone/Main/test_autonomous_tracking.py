from drone_controller import DroneController
from video_processor import VideoProcessor
from object_detector import ObjectDetector
from deep_sort_tracker import DeepSortTracker
import time
import threading
import sys
import cv2

def main():
    controller = DroneController()
    
    if not controller.connect():
        print("Exiting due to connection failure")
        sys.exit(1)
    
    battery = controller.get_battery()
    if battery is None or battery < 20:
        print("Battery too low or unavailable, exiting")
        controller.disconnect()
        sys.exit(1)
    
    status_thread = threading.Thread(target=controller.monitor_status, args=(5,))
    status_thread.daemon = True
    status_thread.start()
    
    video_processor = VideoProcessor(controller)
    detector = ObjectDetector()
    tracker = DeepSortTracker()
    
    time.sleep(2)
    if not video_processor.start_stream():
        print("Video stream failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    time.sleep(2)
    if not controller.takeoff():
        print("Takeoff failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    
    print("Tracking objects for 10 seconds. Press 'q' to stop early.")
    start_time = time.time()
    try:
        while time.time() - start_time < 10:
            ret, frame = video_processor.video_capture.read()
            if not ret:
                print("Video frame failed, landing drone")
                break
            annotated_frame, detections = detector.detect_objects(frame)
            success, track_id = tracker.track_object(detections, controller, frame)
            if success and track_id:
                # Annotate track ID
                for d in detections:
                    if d["label"].startswith("person"):
                        x1, y1, x2, y2 = d["bbox"]
                        cv2.putText(
                            annotated_frame, f"ID: {track_id}",
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                        )
            cv2.imshow("Tello Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted, landing drone")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("User interrupted, landing drone")
    
    time.sleep(1)
    controller.tello.send_rc_control(0, 0, 0, 0)
    if not controller.land():
        print("Landing failed, attempting emergency stop")
        controller.emergency_stop()
    
    video_processor.stop_stream()
    controller.get_battery()
    controller.disconnect()

if __name__ == "__main__":
    main()