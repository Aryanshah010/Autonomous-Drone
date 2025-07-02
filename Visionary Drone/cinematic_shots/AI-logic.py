import numpy as np

class FollowMeCinematicAI:
    """
    Rule-based AI for cinematic 'follow me' shot:
    - Keeps subject centered
    - Adjusts speed based on subject movement
    - Hovers if subject stops
    - Adjusts distance for full-body framing
    - Moves down if person sits
    """
    def __init__(self, frame_width, frame_height, target_bbox_width=200, min_bbox_height=100):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_bbox_width = target_bbox_width  # Desired width of bbox for full-body
        self.min_bbox_height = min_bbox_height      # Threshold for detecting sitting
        self.prev_bbox = None
        self.prev_time = None

    def get_control(self, bbox, dt):
        """
        Given the current bounding box [x1, y1, w, h] and time delta, return (forward, up, yaw) velocities.
        """
        if bbox is None:
            return 0, 0, 0  # No person detected

        x1, y1, w, h = bbox
        bbox_center_x = x1 + w / 2
        frame_center_x = self.frame_width / 2
        error_yaw = frame_center_x - bbox_center_x
        yaw_velocity = int(np.clip(error_yaw * 0.1, -30, 30))

        # Distance control: keep person at target size
        error_dist = self.target_bbox_width - w
        forward_velocity = int(np.clip(error_dist * 0.1, -20, 20))

        # Detect subject movement speed (for dynamic speed adjustment)
        speed_factor = 1
        if self.prev_bbox is not None and dt > 0:
            prev_x1, prev_y1, prev_w, prev_h = self.prev_bbox
            dx = (x1 + w/2) - (prev_x1 + prev_w/2)
            dy = (y1 + h/2) - (prev_y1 + prev_h/2)
            movement = np.sqrt(dx**2 + dy**2) / dt
            # If subject moves fast, increase speed
            speed_factor = np.clip(movement / 50, 0.5, 2.0)
            forward_velocity = int(forward_velocity * speed_factor)
            yaw_velocity = int(yaw_velocity * speed_factor)

        # If subject stops, drone should hover (no forward movement)
        if self.prev_bbox is not None and dt > 0:
            prev_x1, prev_y1, prev_w, prev_h = self.prev_bbox
            dx = (x1 + w/2) - (prev_x1 + prev_w/2)
            dy = (y1 + h/2) - (prev_y1 + prev_h/2)
            movement = np.sqrt(dx**2 + dy**2) / dt
            if movement < 5:  # Threshold for 'stopped'
                forward_velocity = 0

        # If bounding box is small, move back for full body; if large, move closer
        # (Already handled by error_dist logic)

        # If person is sitting (bbox height much less than width), move down
        up_velocity = 0
        aspect_ratio = h / w if w > 0 else 1
        if h < self.min_bbox_height or aspect_ratio < 1.0:
            up_velocity = -15  # Move down
        elif aspect_ratio > 2.0:
            up_velocity = 10   # Move up if person stands up suddenly

        self.prev_bbox = bbox
        return forward_velocity, up_velocity, yaw_velocity
