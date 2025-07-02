import time
import traceback
import config
from drone_controller import DroneController
from video_processor import VideoProcessor
from object_detector import ObjectDetector 
from deep_sort_tracker import DeepSortTracker
import cv2
import importlib.util
import os

# Dynamically import FollowMeCinematicAI from AI-logic.py
cinematic_ai_path = os.path.join(os.path.dirname(__file__), 'cinematic_shots', 'AI-logic.py')
spec = importlib.util.spec_from_file_location('AIlogic', cinematic_ai_path)
ai_logic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_logic)
FollowMeCinematicAI = ai_logic.FollowMeCinematicAI


def main():
    print("[AutoStart] Waiting for drone to become available...")
    drone = DroneController()
    detector = ObjectDetector()
    video = VideoProcessor(drone)
    tracker = DeepSortTracker()
    cinematic_ai = FollowMeCinematicAI(config.RESOLUTION[0], config.RESOLUTION[1])

    # Try to connect until successful
    while not drone.connect():
        print("[AutoStart] Drone not available, retrying in 5 seconds...")
        time.sleep(5)

    try:
        battery = drone.get_battery()
        if battery is not None and battery < 15:
            print("[AutoStart] Battery too low to fly. Exiting.")
            return

        if not video.start_stream():
            print("[AutoStart] Could not start video stream. Exiting.")
            return

        if not drone.takeoff():
            print("[AutoStart] Takeoff failed. Exiting.")
            return

        print("[AutoStart] Drone is airborne. Starting autonomous tracking...")
        prev_time = time.time()
        tracked_id = None  # The ID of the person to follow

        while True:
            ret, frame = video.video_capture.read()
            if not ret or frame is None:
                print("[AutoStart] Failed to read video frame. Landing and exiting.")
                break

            frame = cv2.resize(frame, config.RESOLUTION)
            _, detections = detector.detect_objects(frame)

            # Use DeepSortTracker to get track IDs and bbox
            found, track_id, tracked_bbox = tracker.track_object(detections, drone, frame, return_bbox=True)

            # If we don't have a tracked_id yet, set it to the first detected person
            if tracked_id is None and found and track_id is not None:
                tracked_id = track_id

            # Find the bbox for the tracked_id
            person_bbox = None
            if found and track_id == tracked_id and tracked_bbox is not None:
                person_bbox = tracked_bbox  # [x1, y1, w, h]

            # Cinematic AI control
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            forward, up, yaw = cinematic_ai.get_control(person_bbox, dt)
            drone.drone.send_rc_control(0, forward, up, yaw)

            # Annotate frame (optional)
            if config.DEBUG:
                for d in detections:
                    x1, y1, w, h = d["bbox"]
                    label = d["label"]
                    conf = d["confidence"]
                    color = (0, 255, 0) if label.startswith("person") else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow("Auto Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[AutoStart] 'q' pressed. Landing and exiting.")
                    break

            # Safety: check battery
            battery = drone.get_battery()
            if battery is not None and battery < 15:
                print("[AutoStart] Battery low. Landing and exiting.")
                break

    except Exception as e:
        print("[AutoStart] Exception occurred:", str(e))
        traceback.print_exc()
    finally:
        print("[AutoStart] Landing and cleaning up...")
        drone.land()
        video.stop_stream()
        cv2.destroyAllWindows()
        drone.disconnect()
        print("[AutoStart] Exited.")


if __name__ == "__main__":
    main() 