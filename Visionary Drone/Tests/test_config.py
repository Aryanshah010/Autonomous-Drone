import os
import config

# Print all settings from config.py
# print("Drone IP:", config.DRONE_IP)
print("Command Port:", config.COMMAND_PORT)
print("Video Port:", config.VIDEO_PORT)
print("State Port:", config.STATE_PORT)
print("Frame Rate:", config.FRAME_RATE)
print("Resolution:", config.RESOLUTION)
print("YOLOv8 Model Path:", config.MODEL_PATH)
print("Confidence Threshold:", config.CONFIDENCE_THRESHOLD)
print("Deep SORT Max Age:", config.DEEP_SORT_MAX_AGE)
print("Deep SORT IoU Threshold:", config.DEEP_SORT_IOU_THRESHOLD)
print("PID Gains:", config.PID_GAINS)
print("Debug:", config.DEBUG)

# Check if YOLOv8 model file exists
if os.path.exists(config.MODEL_PATH):
    print("YOLOv8 model file found at:", config.MODEL_PATH)
else:
    print("Error: YOLOv8 model file not found at:", config.MODEL_PATH)