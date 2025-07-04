import cv2
import time
import threading
import av
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import config
from collections import deque
from cinematic_shots import DroneAI

# Set up the video stream URL from the drone
device_url = f"udp://@0.0.0.0:{config.VIDEO_PORT}"

# Threaded frame grabber for low-latency video
class FrameGrabber(threading.Thread):
    def __init__(self):
        super().__init__()
        self.container = av.open(device_url)
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

    def run(self):
        try:
            for packet in self.container.demux(video=0):
                if self.stopped:
                    break
                for frame in packet.decode():
                    if frame is None:
                        continue
                    try:
                        img = frame.to_ndarray(format='bgr24')
                        with self.lock:
                            self.frame = img
                    except Exception as e:
                        print(f"[FRAME ERROR] {e}")
        except Exception as e:
            print(f"[STREAM ERROR] {e}")

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.container.close()
        except:
            pass

# Appearance database for re-identification
appearance_db = {}  # track_id: histogram
wrist_history = {}

# Extract color histogram from a bounding box region
def extract_histogram(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"[WARNING] Empty crop at {bbox}, skipping histogram.")
        return None
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    except Exception as e:
        print(f"[HISTOGRAM ERROR] {e}")
        return None

# Match a histogram to the appearance database
def match_histogram(hist, db, threshold=0.5):
    best_id, best_score = None, float('inf')
    for tid, ref_hist in db.items():
        score = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
        if score < threshold and score < best_score:
            best_score, best_id = score, tid
    return best_id

# Detect crossed hands gesture using keypoints
def is_crossed(track_id, keypoints, dist_thresh=50):
    pts = np.array(keypoints)
    if len(pts) < 11:
        return False
    ls, rs = pts[5], pts[6]
    lw, rw = pts[9], pts[10]
    dist1 = np.linalg.norm(lw - rs)
    dist2 = np.linalg.norm(rw - ls)
    return dist1 < dist_thresh and dist2 < dist_thresh

def main():
    tello = Tello()
    print("[INFO] Connecting to Tello drone...")
    tello.connect()
    print(f"[INFO] Battery: {tello.get_battery()}%")
    tello.streamon()
    print("[INFO] Video stream started.")

    print("[INFO] Loading YOLO model...")
    model = YOLO(config.MODEL_PATH)
    model.classes = [0]

    deepsort = DeepSort(max_age=50, n_init=3)

    grabber = FrameGrabber()
    grabber.start()
    time.sleep(2)

    frame = grabber.get_frame()
    while frame is None:
        frame = grabber.get_frame()
        time.sleep(0.05)

    frame_h, frame_w = frame.shape[:2]
    ai_controller = DroneAI(tello, (frame_w, frame_h))

    crossed_ids = set()
    id_mapping = {}
    following = False
    target_id = None

    print("[INFO] Taking off...")
    tello.takeoff()
    print("[INFO] Drone has taken off. Waiting for crossed hands gesture to start following.")

    try:
        while True:
            frame = grabber.get_frame()
            if frame is None:
                print("[INFO] Frame is None. Skipping.")
                time.sleep(0.01)
                continue

            results = model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
            keypoints_all = results.keypoints.xy if results.keypoints is not None else []

            detections = []
            for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
                x1, y1, x2, y2 = map(float, box.tolist())
                detections.append([[x1, y1, x2, y2], float(conf), "person"])

            tracks = deepsort.update_tracks(detections, frame=frame)

            for track, keypoints in zip(tracks, keypoints_all):
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                x1, y1, x2, y2 = track.to_ltrb()

                hist = extract_histogram(frame, (x1, y1, x2, y2))
                if hist is None:
                    continue

                matched_id = match_histogram(hist, appearance_db)
                if matched_id is not None and tid not in id_mapping:
                    id_mapping[tid] = matched_id
                    print(f"[RE-ID] Matched Track {tid} to Previous ID {matched_id}")

                if tid not in appearance_db:
                    appearance_db[tid] = hist

                for x, y in keypoints:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                if tid not in crossed_ids and is_crossed(tid, keypoints):
                    print(f"[GESTURE] Crossed Hands Detected | ID: {tid}")
                    crossed_ids.add(tid)
                    if not following:
                        appearance_db.clear()
                        wrist_history.clear()
                        crossed_ids.clear()
                        id_mapping.clear()
                        following = True
                        target_id = tid
                        print(f"[INFO] Now following person with ID: {tid}")

            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                mapped_id = id_mapping.get(tid, tid)
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                color = (0, 0, 255) if tid in crossed_ids else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{mapped_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if following and target_id is not None:
                target_track = next((trk for trk in tracks if trk.is_confirmed() and trk.track_id == target_id), None)
                if target_track:
                    sx, sy, sz = ai_controller.track_target(target_track)
                    print(f"[AI ACTION] Moving | X: {sx}, Y: {sy}, Z: {sz}")
                else:
                    print("[WARN] Lost target, hovering.")
                    ai_controller.hover()
            else:
                ai_controller.hover()

            cv2.imshow("Pose + DeepSORT + ReID + Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting...")
                break

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        grabber.stop()
        try:
            tello.streamoff()
        except:
            pass
        print("[INFO] Landing drone...")
        tello.land()
        print(f"[INFO] Battery after flight: {tello.get_battery()}%")
        tello.end()
        cv2.destroyAllWindows()
        appearance_db.clear()
        wrist_history.clear()
        crossed_ids.clear()
        id_mapping.clear()
        print("[INFO] All tracking data cleared. Program ended.")

if __name__ == '__main__':
    main()