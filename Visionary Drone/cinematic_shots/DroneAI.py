import time
import numpy as np
from collections import deque
import logging

# Set up logging
logger = logging.getLogger(__name__)

class DroneAI:
    def __init__(self, tello, frame_size):
        """Initialize DroneAI with Tello-specific optimizations"""
        try:
            self.tello = tello
            if not frame_size or len(frame_size) != 2:
                raise ValueError("Invalid frame size")
                
            self.frame_w, self.frame_h = frame_size
            
            # Validate frame dimensions
            if self.frame_w <= 0 or self.frame_h <= 0:
                raise ValueError(f"Invalid frame dimensions: {frame_size}")
                
            self.center_x = self.frame_w // 2
            self.center_y = self.frame_h // 2
            
            # Tello-optimized command timing
            self.last_command_time = 0
            self.command_delay = 0.1  # 10 commands/second (Tello maximum)
            self.last_processing_time = 0
            self.processing_interval = 0.1  # Process every 100ms

            # Tello-optimized PID control parameters
            self.kp_x, self.ki_x, self.kd_x = 0.12, 0.0005, 0.03  # Conservative gains
            self.kp_y, self.ki_y, self.kd_y = 0.12, 0.0005, 0.03  # Conservative gains
            self.kp_z, self.ki_z, self.kd_z = 0.20, 0.001, 0.05   # Distance control
            self.kp_yaw, self.ki_yaw, self.kd_yaw = 0.10, 0.0005, 0.02  # Gentle rotation

            # PID state variables
            self.integral_x = 0
            self.integral_y = 0
            self.integral_z = 0
            self.integral_yaw = 0
            self.prev_error_x = 0
            self.prev_error_y = 0
            self.prev_error_z = 0
            self.prev_error_yaw = 0

            # Adaptive thresholds based on target size
            self.base_thresh_x = 25
            self.base_thresh_y = 25
            self.min_box_ratio = 0.04  # Too far
            self.max_box_ratio = 0.20  # Too close
            self.optimal_ratio = 0.12  # Target distance

            # Tello-specific safety parameters
            self.min_battery_threshold = 20  # Land when battery < 20%
            self.max_flight_time = 600  # 10 minutes max flight
            self.flight_start_time = time.time()
            self.last_battery_check = 0
            self.battery_check_interval = 30  # Check battery every 30 seconds
            self.emergency_landing = False

            # Movement history for prediction
            self.position_history = deque(maxlen=10)
            
            # Tello-optimized speed limits
            self.max_speed = 50  # Conservative max speed
            self.min_speed = 10  # Responsive min speed
            
            # Smoothing factors
            self.smoothing_factor = 0.7
            self.last_speeds = [0, 0, 0, 0]  # [x, y, z, yaw]

            # Performance monitoring
            self.command_times = deque(maxlen=50)
            self.last_command_sent = 0

            # Tracking state
            self.target_lost_count = 0
            self.max_target_lost = 10  # Frames before considering target lost
            self.last_target_position = None
            
            logger.info("DroneAI initialized successfully for Tello")
            
        except Exception as e:
            logger.error(f"DroneAI initialization error: {e}")
            raise

    def calculate_adaptive_thresholds(self, area_ratio):
        """Dynamically adjust thresholds based on target distance with validation"""
        try:
            if area_ratio is None or np.isnan(area_ratio) or np.isinf(area_ratio):
                logger.warning("Invalid area_ratio provided, using default thresholds")
                return self.base_thresh_x, self.base_thresh_y
                
            if area_ratio < 0:
                logger.warning("Negative area_ratio detected, using default thresholds")
                return self.base_thresh_x, self.base_thresh_y
                
            # Prevent division by zero
            distance_factor = 1.0 / (area_ratio + 0.01)  # Closer = smaller thresholds
            
            # Validate distance factor
            if np.isnan(distance_factor) or np.isinf(distance_factor):
                logger.warning("Invalid distance factor calculated, using default thresholds")
                return self.base_thresh_x, self.base_thresh_y
                
            thresh_x = max(15, min(40, self.base_thresh_x * distance_factor))
            thresh_y = max(15, min(40, self.base_thresh_y * distance_factor))
            
            return thresh_x, thresh_y
            
        except Exception as e:
            logger.error(f"Adaptive threshold calculation error: {e}")
            return self.base_thresh_x, self.base_thresh_y

    def check_tello_safety(self):
        """Check Tello-specific safety conditions"""
        try:
            current_time = time.time()
            
            # Check if emergency landing is needed
            if self.emergency_landing:
                return False
            
            # Check battery periodically
            if current_time - self.last_battery_check > self.battery_check_interval:
                try:
                    battery = self.tello.get_battery()
                    if battery < self.min_battery_threshold:
                        logger.warning(f"Low battery: {battery}%. Initiating emergency landing.")
                        self.emergency_landing = True
                        self.tello.land()
                        return False
                    elif battery < 30:
                        logger.warning(f"Battery getting low: {battery}%")
                except Exception as e:
                    logger.error(f"Battery check failed: {e}")
                self.last_battery_check = current_time
            
            # Check flight time
            if current_time - self.flight_start_time > self.max_flight_time:
                logger.warning("Maximum flight time reached. Landing.")
                self.emergency_landing = True
                self.tello.land()
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety check error: {e}")
            return False

    def safe_mean(self, values):
        """Calculate mean safely for potentially empty lists"""
        try:
            if not values:
                return 0.0
            result = np.mean(values)
            return 0.0 if np.isnan(result) or np.isinf(result) else result
        except Exception as e:
            logger.error(f"Safe mean calculation error: {e}")
            return 0.0

    def predict_target_movement(self, current_pos, current_time):
        """Predict where the target will be based on movement history with validation"""
        try:
            if current_pos is None or len(current_pos) < 2:
                return current_pos
                
            if len(self.position_history) < 3:
                return current_pos
                
            # Calculate velocity from recent positions
            recent_positions = list(self.position_history)[-3:]
            velocities = []
            
            for i in range(1, len(recent_positions)):
                try:
                    dx = recent_positions[i][0] - recent_positions[i-1][0]
                    dy = recent_positions[i][1] - recent_positions[i-1][1]
                    dt = recent_positions[i][2] - recent_positions[i-1][2]
                    
                    # Validate time delta
                    if dt > 0 and not np.isnan(dt) and not np.isinf(dt):
                        # Validate velocity components
                        if not np.isnan(dx) and not np.isnan(dy) and not np.isinf(dx) and not np.isinf(dy):
                            velocities.append([dx/dt, dy/dt])
                        else:
                            logger.warning("Invalid velocity components detected")
                    elif dt <= 0:
                        logger.warning("Non-positive time delta detected")
                    else:
                        logger.warning("Invalid time delta detected")
                        
                except Exception as e:
                    logger.warning(f"Velocity calculation error: {e}")
                    continue
            
            if not velocities:
                return current_pos
                
            # Use safe mean calculation
            avg_vx = self.safe_mean([v[0] for v in velocities])
            avg_vy = self.safe_mean([v[1] for v in velocities])
            
            # Predict position 0.2 seconds ahead
            prediction_time = 0.2
            predicted_x = current_pos[0] + avg_vx * prediction_time
            predicted_y = current_pos[1] + avg_vy * prediction_time
            
            # Validate predicted position
            if np.isnan(predicted_x) or np.isnan(predicted_y) or np.isinf(predicted_x) or np.isinf(predicted_y):
                logger.warning("Invalid predicted position calculated")
                return current_pos
                
            return [predicted_x, predicted_y, current_time]
                
        except Exception as e:
            logger.error(f"Movement prediction error: {e}")
            return current_pos

    def calculate_dynamic_speeds(self, error_x, error_y, error_z, error_yaw, area_ratio):
        """Calculate speeds using PID control with Tello-specific limits"""
        try:
            # Validate input parameters
            errors = [error_x, error_y, error_z, error_yaw]
            for i, error in enumerate(errors):
                if error is None or np.isnan(error) or np.isinf(error):
                    logger.warning(f"Invalid error value at index {i}: {error}")
                    errors[i] = 0
                    
            error_x, error_y, error_z, error_yaw = errors
            
            # Validate area_ratio
            if area_ratio is None or np.isnan(area_ratio) or np.isinf(area_ratio):
                logger.warning("Invalid area_ratio, using default value")
                area_ratio = self.optimal_ratio
            
            # Update integral terms with windup protection
            self.integral_x += error_x
            self.integral_y += error_y
            self.integral_z += error_z
            self.integral_yaw += error_yaw
            
            # Limit integral windup with validation
            integral_limits = [(-100, 100), (-100, 100), (-100, 100), (-50, 50)]
            integrals = [self.integral_x, self.integral_y, self.integral_z, self.integral_yaw]
            
            for i, (integral, (min_val, max_val)) in enumerate(zip(integrals, integral_limits)):
                if np.isnan(integral) or np.isinf(integral):
                    logger.warning(f"Invalid integral value at index {i}, resetting to 0")
                    integrals[i] = 0
                else:
                    integrals[i] = np.clip(integral, min_val, max_val)
                    
            self.integral_x, self.integral_y, self.integral_z, self.integral_yaw = integrals
            
            # Calculate derivatives with validation
            derivative_x = error_x - self.prev_error_x
            derivative_y = error_y - self.prev_error_y
            derivative_z = error_z - self.prev_error_z
            derivative_yaw = error_yaw - self.prev_error_yaw
            
            # Validate derivatives
            derivatives = [derivative_x, derivative_y, derivative_z, derivative_yaw]
            for i, derivative in enumerate(derivatives):
                if np.isnan(derivative) or np.isinf(derivative):
                    logger.warning(f"Invalid derivative at index {i}, setting to 0")
                    derivatives[i] = 0
                    
            derivative_x, derivative_y, derivative_z, derivative_yaw = derivatives
            
            # PID control calculations
            try:
                pid_x = (self.kp_x * error_x + 
                        self.ki_x * self.integral_x + 
                        self.kd_x * derivative_x)
                
                pid_y = (self.kp_y * error_y + 
                        self.ki_y * self.integral_y + 
                        self.kd_y * derivative_y)
                
                # Distance control with smooth transitions
                distance_error = area_ratio - self.optimal_ratio
                pid_z = (self.kp_z * distance_error + 
                        self.ki_z * self.integral_z + 
                        self.kd_z * derivative_z)
                
                # YAW control for rotation
                pid_yaw = (self.kp_yaw * error_yaw + 
                          self.ki_yaw * self.integral_yaw + 
                          self.kd_yaw * derivative_yaw)
                          
            except Exception as e:
                logger.error(f"PID calculation error: {e}")
                pid_x = pid_y = pid_z = pid_yaw = 0
            
            # Validate PID outputs
            pid_outputs = [pid_x, pid_y, pid_z, pid_yaw]
            for i, pid_output in enumerate(pid_outputs):
                if np.isnan(pid_output) or np.isinf(pid_output):
                    logger.warning(f"Invalid PID output at index {i}, setting to 0")
                    pid_outputs[i] = 0
                    
            pid_x, pid_y, pid_z, pid_yaw = pid_outputs
            
            # Dynamic speed limits based on error magnitude
            try:
                error_magnitude = np.sqrt(error_x**2 + error_y**2)
                if np.isnan(error_magnitude) or np.isinf(error_magnitude):
                    error_magnitude = 0
                    
                dynamic_max_speed = min(self.max_speed, 
                                       self.min_speed + error_magnitude * 0.5)
                                       
            except Exception as e:
                logger.error(f"Dynamic speed calculation error: {e}")
                dynamic_max_speed = self.max_speed
            
            # Apply Tello-specific speed limits
            speed_x = int(np.clip(pid_x, -dynamic_max_speed, dynamic_max_speed))
            speed_y = int(np.clip(pid_y, -dynamic_max_speed, dynamic_max_speed))
            speed_z = int(np.clip(pid_z, -dynamic_max_speed, dynamic_max_speed))
            speed_yaw = int(np.clip(pid_yaw, -20, 20))  # Conservative yaw limit
            
            # Update previous errors
            self.prev_error_x = error_x
            self.prev_error_y = error_y
            self.prev_error_z = distance_error
            self.prev_error_yaw = error_yaw
            
            return speed_x, speed_y, speed_z, speed_yaw
            
        except Exception as e:
            logger.error(f"Dynamic speed calculation error: {e}")
            return 0, 0, 0, 0

    def apply_speed_smoothing(self, new_speeds):
        """Apply smoothing to prevent jerky movements with validation"""
        try:
            if new_speeds is None or len(new_speeds) != 4:
                logger.warning("Invalid new_speeds provided")
                return self.last_speeds
                
            smoothed_speeds = []
            for i, new_speed in enumerate(new_speeds):
                try:
                    if np.isnan(new_speed) or np.isinf(new_speed):
                        logger.warning(f"Invalid new_speed at index {i}, using previous value")
                        smoothed_speeds.append(self.last_speeds[i])
                    else:
                        smoothed = (self.smoothing_factor * self.last_speeds[i] + 
                                   (1 - self.smoothing_factor) * new_speed)
                        smoothed_speeds.append(int(smoothed))
                        self.last_speeds[i] = smoothed
                except Exception as e:
                    logger.warning(f"Speed smoothing error at index {i}: {e}")
                    smoothed_speeds.append(self.last_speeds[i])
                    
            return smoothed_speeds
            
        except Exception as e:
            logger.error(f"Speed smoothing error: {e}")
            return self.last_speeds

    def get_movement_description(self, speed_x, speed_y, speed_z, speed_yaw):
        """Convert speed values to human-readable movement descriptions"""
        try:
            movements = []
            
            speeds = [speed_x, speed_y, speed_z, speed_yaw]
            directions = ["right", "up", "forward", "clockwise"]
            opposites = ["left", "down", "backward", "counter-clockwise"]
            
            for i, speed in enumerate(speeds):
                if abs(speed) > 0:
                    direction = directions[i] if speed > 0 else opposites[i]
                    intensity = "slowly" if abs(speed) < 8 else "moderately" if abs(speed) < 12 else "quickly"
                    movements.append(f"{intensity} {direction}")
            
            if not movements:
                return "staying still"
            
            return " and ".join(movements)
            
        except Exception as e:
            logger.error(f"Movement description error: {e}")
            return "error in movement"

    def get_distance_description(self, area_ratio):
        """Convert area ratio to human-readable distance description"""
        try:
            if area_ratio is None or np.isnan(area_ratio) or np.isinf(area_ratio):
                return "invalid distance"
                
            if area_ratio < self.min_box_ratio:
                return "too far away"
            elif area_ratio > self.max_box_ratio:
                return "too close"
            else:
                return "at optimal distance"
                
        except Exception as e:
            logger.error(f"Distance description error: {e}")
            return "error in distance"

    def monitor_performance(self):
        """Monitor command execution and response times"""
        try:
            if len(self.command_times) > 0:
                avg_command_time = np.mean(self.command_times)
                if not np.isnan(avg_command_time) and not np.isinf(avg_command_time):
                    logger.info(f"Performance: Avg command time: {avg_command_time:.3f}s")
                else:
                    logger.warning("Invalid average command time calculated")
            else:
                logger.info("Performance: No command times recorded")
                
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")

    def track_target(self, track):
        """Advanced target tracking with Tello-specific safety checks"""
        try:
            current_time = time.time()
            
            # Check Tello safety conditions
            if not self.check_tello_safety():
                return 0, 0, 0, 0
            
            # Check processing interval
            if current_time - self.last_processing_time < self.processing_interval:
                return 0, 0, 0, 0
            
            self.last_processing_time = current_time
            
            # Validate track object
            if track is None:
                logger.warning("Invalid track object provided")
                return 0, 0, 0, 0
            
            # Get bounding box and calculate center
            try:
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                
                # Validate bounding box
                if x2 <= x1 or y2 <= y1:
                    logger.warning("Invalid bounding box dimensions")
                    return 0, 0, 0, 0
                    
                cx = (x1 + x2) / 2.0  # Use float division for precision
                cy = (y1 + y2) / 2.0
                
                # Validate center coordinates
                if cx < 0 or cy < 0 or cx >= self.frame_w or cy >= self.frame_h:
                    logger.warning(f"Center coordinates out of bounds: ({cx}, {cy})")
                    return 0, 0, 0, 0
                    
            except Exception as e:
                logger.error(f"Bounding box processing error: {e}")
                return 0, 0, 0, 0
            
            # Update target position for tracking validation
            self.last_target_position = (cx, cy)
            self.target_lost_count = 0
            
            # Update position history for prediction
            self.position_history.append([cx, cy, current_time])
            
            # Predict target movement
            predicted_pos = self.predict_target_movement([cx, cy, current_time], current_time)
            target_cx, target_cy = predicted_pos[0], predicted_pos[1]
            
            # Calculate errors from predicted position
            error_x = target_cx - self.center_x
            error_y = self.center_y - target_cy  # Inverted for drone control
            
            # Calculate YAW error (horizontal angle) with safety check
            try:
                if self.frame_w > 0:
                    yaw_error = np.arctan2(error_x, self.frame_w/2) * 180 / np.pi
                    if np.isnan(yaw_error) or np.isinf(yaw_error):
                        yaw_error = 0
                else:
                    yaw_error = 0
            except Exception as e:
                logger.warning(f"YAW calculation error: {e}")
                yaw_error = 0
            
            # Calculate distance metrics
            try:
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = self.frame_w * self.frame_h
                
                # Validate areas
                if frame_area <= 0:
                    logger.error("Invalid frame area")
                    return 0, 0, 0, 0
                    
                area_ratio = box_area / frame_area
                
                # Validate area ratio
                if np.isnan(area_ratio) or np.isinf(area_ratio) or area_ratio < 0:
                    logger.warning("Invalid area ratio calculated")
                    area_ratio = self.optimal_ratio
                    
            except Exception as e:
                logger.error(f"Distance calculation error: {e}")
                area_ratio = self.optimal_ratio
            
            # Adaptive thresholds
            thresh_x, thresh_y = self.calculate_adaptive_thresholds(area_ratio)
            
            # Calculate speeds using PID control (including YAW)
            speed_x, speed_y, speed_z, speed_yaw = self.calculate_dynamic_speeds(
                error_x, error_y, 0, yaw_error, area_ratio)
            
            # Apply smoothing
            smoothed_speeds = self.apply_speed_smoothing([speed_x, speed_y, speed_z, speed_yaw])
            speed_x, speed_y, speed_z, speed_yaw = smoothed_speeds
            
            # Send command with Tello-optimized throttling
            if current_time - self.last_command_time > self.command_delay:
                command_start = time.time()
                
                try:
                    self.tello.send_rc_control(speed_x, speed_z, speed_y, speed_yaw)
                    command_end = time.time()
                    command_time = command_end - command_start
                    self.command_times.append(command_time)
                    self.last_command_sent = command_end
                    
                except Exception as e:
                    logger.error(f"Drone Error: Command failed: {e}")
                    # Fallback to hover
                    try:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    except Exception as hover_error:
                        logger.error(f"Drone Error: Hover command also failed: {hover_error}")
                
                self.last_command_time = current_time
            
            return speed_x, speed_y, speed_z, speed_yaw
            
        except Exception as e:
            logger.error(f"Target tracking error: {e}")
            return 0, 0, 0, 0

    def hover(self):
        """Smooth hover with gradual speed reduction and error handling"""
        try:
            current_time = time.time()
            
            if current_time - self.last_command_time > self.command_delay:
                # Gradually reduce speeds to zero
                gradual_speeds = []
                for speed in self.last_speeds:
                    try:
                        gradual_speed = int(speed * 0.8)
                        gradual_speeds.append(gradual_speed)
                    except Exception as e:
                        logger.warning(f"Speed reduction error: {e}")
                        gradual_speeds.append(0)
                        
                self.last_speeds = gradual_speeds
                
                if all(abs(speed) < 2 for speed in gradual_speeds):
                    logger.debug("Drone Action: Hovering: maintaining stable position")
                    try:
                        self.tello.send_rc_control(0, 0, 0, 0)
                    except Exception as e:
                        logger.error(f"Drone Error: Hover command failed: {e}")
                    self.last_speeds = [0, 0, 0, 0]
                else:
                    movement_desc = self.get_movement_description(*gradual_speeds)
                    logger.debug(f"Drone Action: Gradual hover: {movement_desc} (reducing speed)")
                    try:
                        self.tello.send_rc_control(*gradual_speeds)
                    except Exception as e:
                        logger.error(f"Drone Error: Gradual hover command failed: {e}")
                
                self.last_command_time = current_time
            else:
                remaining_delay = self.command_delay - (current_time - self.last_command_time)
                logger.debug(f"Drone Command: Hover command throttled: waiting {remaining_delay:.3f} seconds")
                
        except Exception as e:
            logger.error(f"Hover error: {e}")

    def reset_control_state(self):
        """Reset PID state when switching targets with validation"""
        try:
            # Reset PID state variables
            self.integral_x = 0
            self.integral_y = 0
            self.integral_z = 0
            self.integral_yaw = 0
            self.prev_error_x = 0
            self.prev_error_y = 0
            self.prev_error_z = 0
            self.prev_error_yaw = 0
            
            # Clear history
            self.position_history.clear()
            self.last_speeds = [0, 0, 0, 0]
            
            # Reset tracking state
            self.target_lost_count = 0
            self.last_target_position = None
            
            logger.info("Drone System: Control system reset: ready to track new target")
            
        except Exception as e:
            logger.error(f"Control state reset error: {e}")

    def is_target_lost(self):
        """Check if target has been lost for too long"""
        try:
            return self.target_lost_count > self.max_target_lost
        except Exception as e:
            logger.error(f"Target lost check error: {e}")
            return True  # Assume target is lost if check fails

    def update_target_lost_count(self):
        """Update target lost counter with validation"""
        try:
            self.target_lost_count += 1
            
            # Validate counter
            if self.target_lost_count < 0:
                logger.warning("Negative target lost count detected, resetting to 0")
                self.target_lost_count = 0
                
            if self.target_lost_count == 1:
                logger.warning("Target lost, starting counter...")
            elif self.target_lost_count % 5 == 0:
                logger.warning(f"Target lost for {self.target_lost_count} frames")
                
        except Exception as e:
            logger.error(f"Target lost count update error: {e}")