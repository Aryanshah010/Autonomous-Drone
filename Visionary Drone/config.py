# Configuration settings for Tello autonomous drone project

# Drone network settings
DRONE_IP = "192.168.10.1"  # Default IP address for Tello's Wi-Fi hotspot
COMMAND_PORT = 8889  # UDP port for sending commands (e.g., takeoff, streamon)
VIDEO_PORT = 11111  # UDP port for receiving video stream
STATE_PORT = 8890  # UDP port for receiving drone status (e.g., battery)

# Video stream settings
FRAME_RATE = 30  # Frames per second for video processing
RESOLUTION = (960, 720)  # Tello's default video resolution (720p)

# YOLOv8 model settings
MODEL_PATH = "/Users/aryanshah/Developer/Autonomous-Drone/Visionary Drone/models/yolo11m-pose.pt"
CONFIDENCE_THRESHOLD = 0.6  # Increased for faster processing

# Deep SORT tracking settings
DEEP_SORT_MAX_AGE = 30  # Max frames to track an object after it's lost
DEEP_SORT_IOU_THRESHOLD = 2  # Intersection-over-Union threshold for associating detections

# Tello-specific settings
TELLO_COMMAND_DELAY = 0.1  # 10 commands per second (Tello maximum)
TELLO_MAX_SPEED = 50  # Conservative speed limit
TELLO_MIN_SPEED = 10  # Minimum speed for responsiveness
TELLO_BATTERY_THRESHOLD = 20  # Land when battery < 20%
TELLO_MAX_FLIGHT_TIME = 600  # 10 minutes max flight
TELLO_PROCESSING_INTERVAL = 0.1  # Process every 100ms

# PID controller gains optimized for Tello
PID_GAINS = {
    "kp": 0.12,  # Reduced for stability
    "ki": 0.0005,  # Reduced to prevent windup
    "kd": 0.03,  # Reduced for smooth movement
}

# Memory management
MAX_APPEARANCE_DB_SIZE = 20  # Reduced for Tello's processing capabilities

# Debugging settings
DEBUG = False  # Enable/disable video display and logging