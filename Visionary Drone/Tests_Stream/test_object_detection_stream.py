from drone_controller import DroneController
from video_processor import VideoProcessor
from object_detector import ObjectDetector
import time
import threading
import sys
import cv2
import base64
import asyncio
import websockets
import numpy as np

async def send_video_frames(websocket, video_processor, detector):
    """Send video frames with detections to WebSocket clients."""
    try:
        while video_processor.streaming:
            ret, frame = video_processor.video_capture.read()
            if not ret:
                print("Warning: Could not read frame, skipping")
                await asyncio.sleep(0.1)
                continue
            # Detect objects
            annotated_frame, detections = detector.detect_objects(frame)
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            # Convert to base64
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            # Send frame and detections
            data = {"frame": jpg_as_text, "detections": [d["label"] for d in detections]}
            await websocket.send(str(data))
            await asyncio.sleep(0.1)  # Limit to ~10 FPS to reduce load
    except Exception as e:
        print("WebSocket error:", str(e))

async def start_websocket_server(video_processor, detector):
    """Start WebSocket server to stream video."""
    async with websockets.serve(
        lambda ws: send_video_frames(ws, video_processor, detector), 
        "0.0.0.0", 8765
    ):
        await asyncio.Future()  # Run forever

def run_websocket(video_processor, detector):
    """Run WebSocket server in a separate thread."""
    asyncio.run(start_websocket_server(video_processor, detector))

def main():
    # Initialize drone controller
    controller = DroneController()
    
    # Connect to drone
    if not controller.connect():
        print("Exiting due to connection failure")
        sys.exit(1)
    
    # Get initial battery level
    battery = controller.get_battery()
    if battery is None or battery < 20:
        print("Battery too low or unavailable, exiting")
        controller.disconnect()
        sys.exit(1)
    
    # Start status monitoring in a thread
    status_thread = threading.Thread(target=controller.monitor_status, args=(5,))
    status_thread.daemon = True
    status_thread.start()
    
    # Initialize video processor and object detector
    video_processor = VideoProcessor(controller)
    detector = ObjectDetector()
    
    # Start video stream
    time.sleep(2)  # Delay for connection stability
    if not video_processor.start_stream():
        print("Video stream failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    # Start WebSocket server in a thread
    websocket_thread = threading.Thread(target=run_websocket, args=(video_processor, detector))
    websocket_thread.daemon = True
    websocket_thread.start()
    
    # Perform takeoff
    time.sleep(2)  # Delay for stream stability
    if not controller.takeoff():
        print("Takeoff failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    # Display video locally and stream for 5 seconds
    print("Streaming video for 5 seconds. Press 'q' to stop early.")
    start_time = time.time()
    try:
        while time.time() - start_time < 5:
            if not video_processor.display_frame():
                print("Video display failed, landing drone")
                break
            time.sleep(0.1)  # Reduce CPU load
    except KeyboardInterrupt:
        print("User interrupted, landing drone")
    
    # Land the drone
    time.sleep(1)  # Delay to ensure video stops
    if not controller.land():
        print("Landing failed, attempting emergency stop")
        controller.emergency_stop()
    
    # Stop video stream
    video_processor.stop_stream()
    
    # Get final battery level
    controller.get_battery()
    
    # Disconnect
    controller.disconnect()

if __name__ == "__main__":
    main()