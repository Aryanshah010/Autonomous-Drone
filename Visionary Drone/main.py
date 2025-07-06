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
from cinematic_shots.DroneAI import DroneAI
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set up the video stream URL from the drone
device_url = f"udp://@0.0.0.0:{config.VIDEO_PORT}"

# Threaded frame grabber for low-latency video
class FrameGrabber(threading.Thread):
    def __init__(self):
        super().__init__()
        self.container = None
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.fallback_frame = None
        self.last_frame_time = 0
        self.frame_timeout = 5.0  # 5 seconds timeout

    def optimize_for_tello():
        """Optimize processing for Tello's capabilities"""
        # Reduce processing frequency
        PROCESSING_INTERVAL = 0.1  # Process every 100ms instead of every frame
        
        # Reduce YOLO confidence for faster processing
        config.CONFIDENCE_THRESHOLD = 0.6  # Higher threshold = fewer detections
        
        # Limit tracking database size
        MAX_APPEARANCE_DB_SIZE = 20  # Reduced from 50


    def initialize_stream(self):
        """Initialize video stream with error handling"""
        try:
            # Close existing container if any
            if self.container:
                try:
                    self.container.close()
                except:
                    pass
                    
            self.container = av.open(device_url, timeout=10.0, options={
                'rtsp_transport': 'udp',
                'stimeout': '5000000',  # 5 second timeout
                'fflags': 'nobuffer',
                'flags': 'low_delay'
            })
            logger.info("Video stream initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize video stream: {e}")
            return False

    def run(self):
        while not self.stopped and self.connection_attempts < self.max_connection_attempts:
            if not self.initialize_stream():
                self.connection_attempts += 1
                time.sleep(2)
                continue

            try:
                for packet in self.container.demux(video=0):
                    if self.stopped:
                        break
                        
                    try:
                        for frame in packet.decode():
                            if frame is None:
                                continue
                            try:
                                img = frame.to_ndarray(format='bgr24')
                                if img is not None and img.size > 0:
                                    with self.lock:
                                        self.frame = img
                                        self.last_frame_time = time.time()
                                        # Store a fallback frame
                                        if self.fallback_frame is None:
                                            self.fallback_frame = img.copy()
                                else:
                                    logger.warning("Received empty or invalid frame")
                            except Exception as e:
                                logger.error(f"Frame conversion error: {e}")
                                # Use fallback frame if available
                                if self.fallback_frame is not None:
                                    with self.lock:
                                        self.frame = self.fallback_frame.copy()
                    except Exception as e:
                        logger.warning(f"Packet decode error: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Stream error: {e}")
                self.connection_attempts += 1
                time.sleep(1)
                continue

        logger.warning("Frame grabber stopped")

    def get_frame(self):
        """Thread-safe frame retrieval with error handling"""
        with self.lock:
            current_time = time.time()
            
            # Check if frame is too old
            if self.frame is not None and (current_time - self.last_frame_time) > self.frame_timeout:
                logger.warning("Frame is too old, using fallback")
                self.frame = self.fallback_frame.copy() if self.fallback_frame is not None else None
            
            if self.frame is not None:
                try:
                    return self.frame.copy()
                except Exception as e:
                    logger.error(f"Frame copy error: {e}")
                    return self.fallback_frame.copy() if self.fallback_frame is not None else None
            return self.fallback_frame.copy() if self.fallback_frame is not None else None

    def stop(self):
        self.stopped = True
        try:
            if self.container:
                self.container.close()
        except Exception as e:
            logger.error(f"Error closing container: {e}")

# Appearance database for re-identification with memory management
appearance_db = {}  # track_id: histogram
MAX_APPEARANCE_DB_SIZE = 50  # Prevent memory leaks

def manage_appearance_db():
    """Manage appearance database size to prevent memory leaks"""
    if len(appearance_db) > MAX_APPEARANCE_DB_SIZE:
        logger.info(f"Clearing appearance database (size: {len(appearance_db)})")
        appearance_db.clear()


# Add these utility functions at the top of main.py

def safe_zip_detections(boxes, confidences):
    """Safely zip detection boxes and confidences"""
    try:
        if boxes is None or confidences is None:
            return []
        min_len = min(len(boxes), len(confidences))
        return list(zip(boxes[:min_len], confidences[:min_len]))
    except Exception as e:
        logger.error(f"Safe zip detections error: {e}")
        return []

def safe_zip_tracks_keypoints(tracks, keypoints_all):
    """Safely zip tracks and keypoints"""
    try:
        if tracks is None or keypoints_all is None:
            return []
        min_len = min(len(tracks), len(keypoints_all))
        return list(zip(tracks[:min_len], keypoints_all[:min_len]))
    except Exception as e:
        logger.error(f"Safe zip tracks keypoints error: {e}")
        return []

def safe_mean(values):
    """Calculate mean safely for potentially empty lists"""
    try:
        if not values:
            return 0.0
        result = np.mean(values)
        return 0.0 if np.isnan(result) or np.isinf(result) else result
    except Exception as e:
        logger.error(f"Safe mean calculation error: {e}")
        return 0.0
    
# Extract color histogram from a bounding box region
def extract_histogram(image, bbox):
    """Extract histogram with comprehensive error handling"""
    try:
        if image is None or image.size == 0:
            logger.warning("Invalid image provided for histogram extraction")
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # Validate frame dimensions
        if w <= 0 or h <= 0:
            logger.error(f"Invalid frame dimensions: {w}x{h}")
            return None

        # Clamp coordinates to valid range
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Validate crop region
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop region: {bbox}")
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning(f"Empty crop at {bbox}, skipping histogram.")
            return None

        # Convert to HSV and calculate histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        normalized_hist = cv2.normalize(hist, hist).flatten()
        
        # Validate histogram
        if np.any(np.isnan(normalized_hist)) or np.any(np.isinf(normalized_hist)):
            logger.warning("Invalid histogram values detected")
            return None
            
        return normalized_hist

    except Exception as e:
        logger.error(f"Histogram extraction error: {e}")
        return None

# Match a histogram to the appearance database
def match_histogram(hist, db, threshold=0.5):
    """Match histogram with error handling"""
    try:
        if hist is None or db is None or len(db) == 0:
            return None

        best_id, best_score = None, float('inf')
        for tid, ref_hist in db.items():
            try:
                score = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
                if not np.isnan(score) and not np.isinf(score):
                    if score < threshold and score < best_score:
                        best_score, best_id = score, tid
            except Exception as e:
                logger.warning(f"Histogram comparison error for ID {tid}: {e}")
                continue
                
        return best_id
    except Exception as e:
        logger.error(f"Histogram matching error: {e}")
        return None

# Detect crossed hands gesture using keypoints with validation
def is_crossed(keypoints, dist_thresh=50):
    """Detect crossed hands gesture with comprehensive validation"""
    try:
        if keypoints is None:
            return False

        # Fix NumPy deprecation warning
        pts = np.array(keypoints, dtype=np.float64)
        
        # Validate keypoint array
        if len(pts) < 11:
            logger.debug("Insufficient keypoints for gesture detection")
            return False

        # Check for invalid values
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            logger.warning("Invalid keypoint values detected")
            return False

        # Check for negative coordinates
        if np.any(pts < 0):
            logger.warning("Negative keypoint coordinates detected")
            return False

        # Extract relevant keypoints (shoulders and wrists)
        try:
            ls, rs = pts[5], pts[6]  # Left and right shoulders
            lw, rw = pts[9], pts[10]  # Left and right wrists
        except IndexError as e:
            logger.error(f"Keypoint index error: {e}")
            return False

        # Calculate distances with validation
        try:
            dist1 = np.linalg.norm(lw - rs)
            dist2 = np.linalg.norm(rw - ls)
            
            # Validate distance calculations
            if np.isnan(dist1) or np.isnan(dist2) or np.isinf(dist1) or np.isinf(dist2):
                logger.warning("Invalid distance calculations")
                return False
                
            return dist1 < dist_thresh and dist2 < dist_thresh
            
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return False

    except Exception as e:
        logger.error(f"Gesture detection error: {e}")
        return False

# Add this function to handle invalid bounding boxes
def validate_bounding_box(x1, y1, x2, y2, frame_w, frame_h):
    """Validate and fix bounding box coordinates"""
    try:
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y2 = max(y1 + 1, min(y2, frame_h))
        
        # Check if box is valid
        if x2 <= x1 or y2 <= y1:
            return None, None, None, None
            
        return x1, y1, x2, y2
        
    except Exception as e:
        logger.error(f"Bounding box validation error: {e}")
        return None, None, None, None

def main():
    """Main function with comprehensive error handling"""
    tello = None
    grabber = None
    
    try:
        # Initialize drone connection
        tello = Tello()
        logger.info("Connecting to Tello drone...")
        
        try:
            tello.connect()
            battery = tello.get_battery()
            logger.info(f"Battery: {battery}%")
            
            if battery < 20:
                logger.warning("Low battery detected!")
                
        except Exception as e:
            logger.error(f"Failed to connect to drone: {e}")
            return

        # Start video stream
        try:
            tello.streamon()
            logger.info("Video stream started.")
        except Exception as e:
            logger.error(f"Failed to start video stream: {e}")
            return

        # Load YOLO model
        logger.info("Loading YOLO model...")
        try:
            model = YOLO(config.MODEL_PATH)
            model.classes = [0]
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return

        # Initialize tracking
        try:
            deepsort = DeepSort(max_age=50, n_init=3)
        except Exception as e:
            logger.error(f"Failed to initialize DeepSORT: {e}")
            return

        # Initialize frame grabber
        grabber = FrameGrabber()
        grabber.start()
        
        # Wait for first frame
        logger.info("Waiting for video stream...")
        frame = None
        timeout = 30  # 30 second timeout
        start_time = time.time()
        
        while frame is None and (time.time() - start_time) < timeout:
            frame = grabber.get_frame()
            time.sleep(0.1)
            
        if frame is None:
            logger.error("Failed to receive video frame within timeout")
            return

        # Initialize AI controller
        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            logger.error(f"Invalid frame dimensions: {frame_w}x{frame_h}")
            return
            
        ai_controller = DroneAI(tello, (frame_w, frame_h))

        # Initialize tracking state
        crossed_ids = set()
        id_mapping = {}
        following = False
        target_id = None
        frame_count = 0

        # Take off
        logger.info("Taking off...")
        try:
            tello.takeoff()
            logger.info("Drone has taken off. Waiting for crossed hands gesture to start following.")
        except Exception as e:
            logger.error(f"Takeoff failed: {e}")
            return
        
        # Main processing loop with Tello optimizations
        last_processing_time = 0
        processing_interval = 0.1  # Process every 100ms

        # Main processing loop
        while True:
            try:

                current_time = time.time()
                
                # Check processing interval
                if current_time - last_processing_time < processing_interval:
                    time.sleep(0.01)
                    continue
                
                last_processing_time = current_time

                frame = grabber.get_frame()
                if frame is None:
                    logger.warning("Frame is None. Skipping.")
                    time.sleep(0.01)
                    continue

                frame_count += 1
                
                # Process frame with YOLO
                try:
                    results = model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
                    keypoints_all = results.keypoints.xy if results.keypoints is not None else []
                except Exception as e:
                    logger.error(f"YOLO processing error: {e}")
                    continue

                # Prepare detections
                detections = []
                try:
                    safe_detections = safe_zip_detections(results.boxes.xyxy, results.boxes.conf)
                    for box, conf in safe_detections:
                        x1, y1, x2, y2 = map(float, box.tolist())
                        detections.append([[x1, y1, x2, y2], float(conf), "person"])
                except Exception as e:
                    logger.error(f"Detection processing error: {e}")
                    continue

                # Update tracking
                try:
                    tracks = deepsort.update_tracks(detections, frame=frame)
                except Exception as e:
                    logger.error(f"DeepSORT update error: {e}")
                    continue

                # Process each track
                safe_tracks_keypoints = safe_zip_tracks_keypoints(tracks, keypoints_all)

                for track, keypoints in safe_tracks_keypoints:
                    try:
                        if not track.is_confirmed():
                            continue
                            
                        tid = track.track_id
                        
                        # Get and validate bounding box
                        try:
                            x1, y1, x2, y2 = map(int, track.to_ltrb())
                            x1, y1, x2, y2 = validate_bounding_box(x1, y1, x2, y2, frame_w, frame_h)
                            
                            if x1 is None:  # Invalid bounding box
                                continue
                                
                        except Exception as e:
                            logger.error(f"Bounding box processing error: {e}")
                            continue

                        # Extract histogram
                        hist = extract_histogram(frame, (x1, y1, x2, y2))
                        if hist is None:
                            continue

                        # Match with appearance database
                        matched_id = match_histogram(hist, appearance_db)
                        if matched_id is not None and tid not in id_mapping:
                            id_mapping[tid] = matched_id
                            logger.info(f"Re-ID: Matched Track {tid} to Previous ID {matched_id}")

                        # Update appearance database
                        if tid not in appearance_db:
                            appearance_db[tid] = hist
                            manage_appearance_db()  # Prevent memory leaks

                        # Draw keypoints
                        try:
                            for x, y in keypoints:
                                # Fix NumPy deprecation warning
                                x_val = float(x) if hasattr(x, '__float__') else x
                                y_val = float(y) if hasattr(y, '__float__') else y
                                
                                if not np.isnan(x_val) and not np.isnan(y_val):
                                    cv2.circle(frame, (int(x_val), int(y_val)), 3, (0, 0, 255), -1)
                        except Exception as e:
                            logger.warning(f"Keypoint drawing error: {e}")

                        # Check for gesture
                        if tid not in crossed_ids and is_crossed(keypoints):
                            logger.info(f"Gesture: Crossed Hands Detected | ID: {tid}")
                            crossed_ids.add(tid)
                            if not following:
                                # Reset tracking state
                                appearance_db.clear()
                                crossed_ids.clear()
                                id_mapping.clear()
                                following = True
                                target_id = tid
                                ai_controller.reset_control_state()
                                logger.info(f"Now following person with ID: {tid}")

                    except Exception as e:
                        logger.error(f"Track processing error: {e}")
                        continue

                # Draw tracking information
                for track in tracks:
                    try:
                        if not track.is_confirmed():
                            continue
                        tid = track.track_id
                        mapped_id = id_mapping.get(tid, tid)
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        color = (0, 0, 255) if tid in crossed_ids else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID:{mapped_id}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        logger.warning(f"Drawing error: {e}")
                        continue

                # Handle target following with safety checks
                if following and target_id is not None:
                    try:
                        target_track = next((trk for trk in tracks 
                                           if trk.is_confirmed() and trk.track_id == target_id), None)
                        if target_track:
                            # Reset target lost counter when target is found
                            ai_controller.target_lost_count = 0
                            sx, sy, sz, syaw = ai_controller.track_target(target_track)
                            logger.info(f"AI Action: Moving | X: {sx}, Y: {sy}, Z: {sz}, YAW: {syaw}")
                        else:
                            # Target lost - update counter and check if should stop following
                            ai_controller.update_target_lost_count()
                            if ai_controller.is_target_lost():
                                logger.warning("Target lost for too long, stopping follow mode")
                                following = False
                                target_id = None
                                ai_controller.hover()
                            else:
                                logger.warning("Target temporarily lost, hovering...")
                                ai_controller.hover()
                    except Exception as e:
                        logger.error(f"Target following error: {e}")
                        ai_controller.hover()
                else:
                    ai_controller.hover()

                # Display frame
                try:
                    cv2.imshow("Pose + DeepSORT + ReID + Gesture", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Quitting...")
                        break
                except Exception as e:
                    logger.error(f"Display error: {e}")

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(0.1)
                continue

    except Exception as e:
        logger.error(f"Critical error: {e}")

    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        if grabber:
            grabber.stop()
            grabber.join(timeout=5)
            
        if tello:
            try:
                tello.streamoff()
                battery = tello.get_battery()
                logger.info(f"Battery after flight: {battery}%")
            except Exception as e:
                logger.error(f"Error getting final battery: {e}")
                
            try:
                logger.info("Landing drone...")
                tello.land()
            except Exception as e:
                logger.error(f"Landing error: {e}")
                
            try:
                tello.end()
            except Exception as e:
                logger.error(f"Drone disconnect error: {e}")

        cv2.destroyAllWindows()
        
        # Clear tracking data
        appearance_db.clear()
        crossed_ids.clear()
        id_mapping.clear()
        
        # Performance summary=
        logger.info("\n" + "="*50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*50)
        if 'ai_controller' in locals():
            ai_controller.monitor_performance()
        logger.info("="*50)
        logger.info("All tracking data cleared. Program ended.")

if __name__ == '__main__':
    main()