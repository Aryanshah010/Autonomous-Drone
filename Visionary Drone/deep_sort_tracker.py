from deep_sort_realtime.deepsort_tracker import DeepSort
import config
import numpy as np

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=config.DEEP_SORT_MAX_AGE,
            nms_max_overlap=config.DEEP_SORT_IOU_THRESHOLD
        )
        self.kp = config.PID_GAINS["kp"]
        self.ki = config.PID_GAINS["ki"]
        self.kd = config.PID_GAINS["kd"]
        self.prev_error_yaw = 0
        self.prev_error_dist = 0
        self.integral_yaw = 0
        self.integral_dist = 0
        self.target_distance = 100  # ~1m in cm
        self.frame_width = config.RESOLUTION[0]
        self.frame_height = config.RESOLUTION[1]

    def compute_pid(self, error, prev_error, integral, kp, ki, kd, dt=0.1):
        integral += error * dt
        derivative = (error - prev_error) / dt
        output = kp * error + ki * integral + kd * derivative
        return output, integral

    def track_object(self, detections, controller, frame, return_bbox=False):
        if not detections or not any(d["label"].startswith("person") for d in detections):
            controller.tello.send_rc_control(0, 0, 0, 0)
            if return_bbox:
                return False, None, None
            return False, None

        # Format detections for Deep SORT: [[x1,y1,x2,y2,confidence], ...]
        deepsort_detections = []
        for d in detections:
            if d["label"].startswith("person"):
                x1, y1, x2, y2 = d["bbox"]
                confidence = d["confidence"]
                deepsort_detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, 0))

        # Update tracker
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        if not tracks:
            controller.tello.send_rc_control(0, 0, 0, 0)
            if return_bbox:
                return False, None, None
            return False, None

        # Select track with highest confidence
        person_track = max(tracks, key=lambda t: t.det_conf if t.det_conf else 0)
        if not person_track.is_confirmed() or person_track.det_class != 0:
            controller.tello.send_rc_control(0, 0, 0, 0)
            if return_bbox:
                return False, None, None
            return False, None

        # Get bounding box
        x1, y1, w, h = person_track.to_tlwh()
        x2, y2 = x1 + w, y1 + h
        track_id = person_track.track_id

        # Compute errors
        bbox_center_x = (x1 + x2) / 2
        frame_center_x = self.frame_width / 2
        error_yaw = frame_center_x - bbox_center_x
        yaw_output, self.integral_yaw = self.compute_pid(
            error_yaw, self.prev_error_yaw, self.integral_yaw, self.kp, self.ki, self.kd
        )
        self.prev_error_yaw = error_yaw

        target_bbox_width = 200  # Calibrated for ~1m
        error_dist = target_bbox_width - w
        dist_output, self.integral_dist = self.compute_pid(
            error_dist, self.prev_error_dist, self.integral_dist, self.kp, self.ki, self.kd
        )
        self.prev_error_dist = error_dist

        # Limit velocities
        yaw_velocity = int(np.clip(yaw_output, -30, 30))
        forward_velocity = int(np.clip(dist_output, -20, 20))

        # Send control command
        controller.tello.send_rc_control(0, forward_velocity, 0, yaw_velocity)
        if return_bbox:
            return True, track_id, [int(x1), int(y1), int(w), int(h)]
        return True, track_id