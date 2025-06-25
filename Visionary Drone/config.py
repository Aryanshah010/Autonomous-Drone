# Configuration settings for Tello autonomous drone project

# Drone network settings
DRONE_IP = "192.168.10.1"  # Default IP address for Tello's Wi-Fi hotspot
COMMAND_PORT = 8889  # UDP port for sending commands (e.g., takeoff, streamon)
VIDEO_PORT = 11111  # UDP port for receiving video stream
STATE_PORT = 8890  # UDP port for receiving drone status (e.g., battery)

# Video stream settings
FRAME_RATE = 30  # Frames per second for video processing
RESOLUTION = (1280, 720)  # Tello's default video resolution (720p)

# YOLOv8 model settings
MODEL_PATH = "models/yolov8n.pt"  # Path to YOLOv8 nano model
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections (0 to 1)

# Deep SORT tracking settings
DEEP_SORT_MAX_AGE = 30  # Max frames to track an object after it's lost
DEEP_SORT_IOU_THRESHOLD = 0.7  # Intersection-over-Union threshold for associating detections

# PID controller gains for autonomous movement
PID_GAINS = {
    "kp": 0.5,  # Proportional gain
    "ki": 0.05,  # Integral gain
    "kd": 0.2,  # Derivative gain
}

# Debugging settings
DEBUG = True  # Enable/disable video display and logging