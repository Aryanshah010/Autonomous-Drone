import time
import numpy as np

class DroneAI:
    def __init__(self, tello, frame_size):
        self.tello = tello
        self.frame_w, self.frame_h = frame_size
        self.center_x = self.frame_w // 2
        self.center_y = self.frame_h // 2
        self.last_command_time = 0
        self.command_delay = 0.1  # seconds

        # PID tuning (Proportional control only)
        self.kp_x = 0.1
        self.kp_y = 0.1
        self.kp_z = 0.2

        # Movement thresholds
        self.thresh_x = 30
        self.thresh_y = 30
        self.min_box_ratio = 0.05  # Too far
        self.max_box_ratio = 0.18  # Too close

    def track_target(self, track):
        # Get bounding box
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Calculate offset from frame center
        offset_x = cx - self.center_x
        offset_y = self.center_y - cy  # Invert y for drone control

        # Calculate bbox area ratio to determine distance
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = self.frame_w * self.frame_h
        area_ratio = box_area / frame_area

        # Initialize speeds
        speed_x = 0  # left/right
        speed_y = 0  # up/down
        speed_z = 0  # forward/backward

        # Determine horizontal adjustment
        if abs(offset_x) > self.thresh_x:
            speed_x = int(np.clip(self.kp_x * offset_x, -20, 20))

        # Determine vertical adjustment
        if abs(offset_y) > self.thresh_y:
            speed_y = int(np.clip(self.kp_y * offset_y, -20, 20))

        # Determine forward/backward adjustment
        if area_ratio < self.min_box_ratio:
            speed_z = 20  # too far
        elif area_ratio > self.max_box_ratio:
            speed_z = -20  # too close
        else:
            speed_z = 0  # maintain distance

        # Throttle command sending to reduce latency issues
        if time.time() - self.last_command_time > self.command_delay:
            self.tello.send_rc_control(speed_x, speed_z, speed_y, 0)
            self.last_command_time = time.time()

        return speed_x, speed_y, speed_z

    def hover(self):
        if time.time() - self.last_command_time > self.command_delay:
            self.tello.send_rc_control(0, 0, 0, 0)
            self.last_command_time = time.time()
