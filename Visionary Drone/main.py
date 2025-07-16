import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# ----------------------------
# Configuration Parameters
# ----------------------------
POSE_MODEL = '/Users/aryanshah/Developer/Autonomous-Drone/Visionary Drone/models/yolo11m-pose.pt'
MAX_FPS = 30
TARGET_AREA = 0.19  # Increased for closer following
AREA_THRESHOLD = 0.005  # More sensitive
DEADZONE_PIX = 25
MAX_SPEED = 65

# PID Gains - Adjusted for better forward/backward control
KP_YAW = 0.6 # Lowered from 0.9 to reduce turbulence
KP_FB = 1.8   # Increased from 0.7 for faster following
KP_UD = 0.3
KP_LR = 0.9

# Tracking parameters
MAX_LOST_FRAMES = 60
REACQUISITION_DISTANCE = 400
HEADING_HISTORY_SIZE = 5

# Movement detection parameters
MOVEMENT_THRESHOLD = 15
HOVER_TIMEOUT = 2.0

# Obstacle detection parameters
OBSTACLE_REGION = (slice(200, 280), slice(280, 360))
OBSTACLE_THRESHOLD = 50

# ----------------------------
# Helper Functions
# ----------------------------

def calculate_person_heading(keypoints):
    """Calculate the heading direction of a person based on keypoints"""
    if keypoints is None or len(keypoints) < 13:
        return None, 0, 0
    
    try:
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        if (len(nose) < 3 or len(left_shoulder) < 3 or len(right_shoulder) < 3 or
            len(left_hip) < 3 or len(right_hip) < 3):
            return None, 0, 0
        
        if (nose[2] < 0.3 or left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3 or
            left_hip[2] < 0.3 or right_hip[2] < 0.3):
            return None, 0, 0
        
        mid_hip_x = (left_hip[0] + right_hip[0]) / 2
        mid_hip_y = (left_hip[1] + right_hip[1]) / 2
        
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        heading_x = shoulder_center_x - mid_hip_x
        heading_y = shoulder_center_y - mid_hip_y
        
        heading_angle = np.degrees(np.arctan2(heading_y, heading_x))
        confidence = (nose[2] + left_shoulder[2] + right_shoulder[2] + left_hip[2] + right_hip[2]) / 5
        
        return (heading_x, heading_y), heading_angle, confidence
        
    except (IndexError, TypeError, ValueError) as e:
        return None, 0, 0

def draw_keypoints_and_heading(frame, keypoints, bbox, heading_info):
    """Draw keypoints and heading direction on the frame"""
    if keypoints is None or len(keypoints) == 0:
        return
    
    for i, kpt in enumerate(keypoints):
        try:
            if len(kpt) >= 3 and kpt[2] > 0.3:
                x, y = int(kpt[0]), int(kpt[1])
                confidence = kpt[2]
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.circle(frame, (x, y), 3, color, -1)
        except (IndexError, TypeError, ValueError):
            continue
    
    if heading_info[0] is not None:
        try:
            heading_vector, angle, confidence = heading_info
            x1, y1, x2, y2 = bbox
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            scale = 50
            end_x = int(center_x + heading_vector[0] * scale)
            end_y = int(center_y + heading_vector[1] * scale)
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 3)
        except (IndexError, TypeError, ValueError):
            pass

def detect_obstacle(frame):
    """Detect obstacles in front of the drone"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        region = gray[OBSTACLE_REGION]
        avg_brightness = np.mean(region)
        return avg_brightness < OBSTACLE_THRESHOLD
    except Exception:
        return False

def detect_movement(current_pos, last_pos, threshold=MOVEMENT_THRESHOLD):
    """Detect if person has moved significantly"""
    if last_pos is None:
        return True
    
    try:
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        return distance > threshold
    except Exception:
        return True

# ----------------------------
# Main Drone Controller
# ----------------------------

def initialize_drone_and_models():
    """Initialize drone connection, video stream, and models"""
    print("Initializing drone connection...")
    tello = Tello()
    tello.connect()
    
    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    
    if battery < 20:
        print("Battery too low to take off. Exiting.")
        return None, None, None, None
    
    print("Starting video stream...")
    tello.streamon()
    time.sleep(0.5)
    
    cap = tello.get_frame_read()
    time.sleep(1)
    
    # Test video stream
    print("Testing video stream...")
    for _ in range(10):
        frame = cap.frame
        if frame is not None:
            h, w, _ = frame.shape
            print(f"Video stream working - Frame size: {w}x{h}")
            break
        time.sleep(0.2)
    else:
        print("Failed to get video stream. Exiting.")
        return None, None, None, None
    
    print("Loading YOLO pose model...")
    try:
        pose_model = YOLO(POSE_MODEL)
        time.sleep(0.5)
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None, None, None, None
    
    print("Initializing DeepSort tracker...")
    try:
        tracker = DeepSort(max_age=30)
        time.sleep(0.5)
        print("DeepSort tracker initialized")
    except Exception as e:
        print(f"Failed to initialize DeepSort: {e}")
        return None, None, None, None
    
    return tello, cap, pose_model, tracker

def test_detection_system(pose_model, cap, duration=3):
    """Test detection system before takeoff"""
    print("Testing detection system...")
    start_time = time.time()
    detection_count = 0
    total_frames = 0
    
    while time.time() - start_time < duration:
        frame = cap.frame
        if frame is not None:
            total_frames += 1
            try:
                results = pose_model(frame, stream=False)[0]
                num_people = len(results.boxes) if results.boxes is not None else 0
                if num_people > 0:
                    detection_count += 1
            except Exception:
                pass
        
        time.sleep(0.1)
    
    detection_rate = detection_count / max(total_frames, 1)
    print(f"Detection test complete - Rate: {detection_rate:.2f}")
    return detection_rate > 0.1

def main():
    tello, cap, pose_model, tracker = initialize_drone_and_models()
    if tello is None:
        print("Initialization failed. Exiting.")
        return

    if not test_detection_system(pose_model, cap):
        print("Detection system not working properly. Exiting.")
        tello.streamoff()
        return

    print("All systems ready. Taking off...")
    try:
        tello.takeoff()
        time.sleep(3)
        print("Takeoff complete")
    except Exception as e:
        print(f"Takeoff failed: {e}")
        tello.streamoff()
        return

    last_time = time.time()
    target_id = None
    target_lost_frames = 0
    last_target_position = None
    hover_mode = False
    
    # --- For adaptive speed ---
    last_cx = None
    last_velocity_time = None
    velocity_smooth = 0

    try:
        while True:
            current_time = time.time()
            if current_time - last_time < 1 / MAX_FPS:
                continue
            last_time = current_time

            frame = cap.frame
            if frame is None:
                time.sleep(0.05)
                continue

            # --- Preprocess frame for YOLO ---
            frame_resized = cv2.resize(frame, (640, 480))
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # --- Run detection ---
            try:
                results = pose_model(frame_resized, stream=False)[0]
            except Exception as e:
                print(f"Detection error: {e}")
                continue

            # --- Find the largest person (by area) ---
            bboxes = []
            person_keypoints = []
            max_area = 0
            target_bbox = None
            for box, kpts in zip(results.boxes.xyxy.tolist(), results.keypoints):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_bbox = (x1, y1, x2, y2)
                    target_kpts = kpts
                bboxes.append(([x1, y1, x2 - x1, y2 - y1], 1.0, 'person'))
                person_keypoints.append(kpts)

            if target_bbox is None:
                print("No person detected.")
                tello.send_rc_control(0, 0, 0, 0)
                continue

            # --- Tracking (optional, for robustness) ---
            tracks = tracker.update_tracks(bboxes, frame=frame_resized)
            # Find the track that matches the largest person
            target_track = None
            for track in tracks:
                if track.is_confirmed():
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    if (x1, y1, x2, y2) == target_bbox:
                        target_track = track
                        break

            # --- Calculate control ---
            x1, y1, x2, y2 = target_bbox
            w, h = 640, 480  # frame_resized shape

            # --- Updated: Use bottom of bbox for vertical control ---
            target_feet_y = int(h * 0.95)  # Show full body, feet near bottom
            feet_y = y2  # Bottom of bounding box
            dy = feet_y - target_feet_y

            # --- Updated: Forward/backward control with new thresholds ---
            area = (x2 - x1) * (y2 - y1) / (w * h)
            area_error = TARGET_AREA - area

            # Center errors (for yaw)
            cx = (x1 + x2) // 2
            dx = cx - w // 2

            # --- Adaptive speed: estimate subject velocity ---
            current_velocity_time = time.time()
            if last_cx is not None and last_velocity_time is not None:
                dt = current_velocity_time - last_velocity_time
                if dt > 0:
                    velocity = (cx - last_cx) / dt  # pixels per second
                    velocity_smooth = 0.8 * velocity_smooth + 0.2 * velocity
                else:
                    velocity = 0
            else:
                velocity = 0
            last_cx = cx
            last_velocity_time = current_velocity_time

            # Use velocity to scale FB (forward/backward) command
            # If subject moves fast, drone follows faster (scale between 1x and 2.5x)
            velocity_factor = min(max(abs(velocity_smooth) / 100, 1), 2.5)

            # Thresholds
            min_yaw = 10   # Increased from 5 to reduce jitter
            min_fb = 8     # (can keep as is, or try 10 for more punch)
            min_ud = 5

            # --- Deadzone for area and feet position ---
            area_deadzone = 0.015  # Only move if area error is > 0.015 (tune as needed)
            feet_deadzone = 20     # Only move if feet error is > 20 pixels (tune as needed)

            # Forward/backward
            fb = 0
            if abs(area_error) > area_deadzone:
                fb = int(np.clip(KP_FB * area_error * MAX_SPEED * 3 * velocity_factor, -MAX_SPEED, MAX_SPEED))
                if abs(fb) < min_fb:
                    fb = min_fb if fb > 0 else -min_fb

            # Up/down (now using feet position, only move up automatically)
            ud = 0
            if dy > feet_deadzone:
                ud = int(np.clip(KP_UD * dy / (h // 2) * MAX_SPEED, 0, MAX_SPEED))
                if ud < min_ud:
                    ud = min_ud
            elif dy < -feet_deadzone:
                ud = 0

            # If both errors are small, don't move
            if abs(area_error) <= area_deadzone and abs(dy) <= feet_deadzone:
                fb = 0
                ud = 0

            # Yaw (left/right rotation)
            yaw = 0
            if abs(dx) > DEADZONE_PIX:
                yaw = int(np.clip(KP_YAW * dx / (w // 2) * MAX_SPEED, -MAX_SPEED, MAX_SPEED))
                if abs(yaw) < min_yaw:
                    yaw = min_yaw if yaw > 0 else -min_yaw

            # Left/right (optional, usually not needed if yaw is used)
            lr = 0

            # --- Send control ---
            tello.send_rc_control(lr, fb, ud, yaw)
            print(f"Controls: LR={lr}, FB={fb}, UD={ud}, Yaw={yaw}, Area={area:.3f}, AreaErr={area_error:.3f}, FeetY={feet_y}, TargetFeetY={target_feet_y}")

            # --- Draw for debug ---
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame_resized, ((x1 + x2) // 2, (y1 + y2) // 2), 5, (0, 0, 255), -1)
            cv2.imshow('Tello Person Follow', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        print("Cleaning up...")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            tello.land()
            time.sleep(1)
            tello.streamoff()
        except Exception:
            pass
        cv2.destroyAllWindows()
        time.sleep(1)

# ----------------------------
# Utility IOU
# ----------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

if __name__ == '__main__':
    main()