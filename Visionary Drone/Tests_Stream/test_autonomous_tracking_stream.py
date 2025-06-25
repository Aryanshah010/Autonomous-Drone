from drone_controller import DroneController
from video_processor import VideoProcessor
from object_detector import ObjectDetector
from deep_sort_tracker import DeepSortTracker
import time
import threading
import sys
import cv2
import base64
import asyncio
import websockets
import json
import numpy as np

async def handle_tracking(websocket, video_processor, detector, tracker, controller):
    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command")
            response = {"command": command}
            if command == "start_stream":
                success = video_processor.start_stream()
                response["status"] = "Streaming" if success else "Failed"
            elif command == "stop_stream":
                video_processor.stop_stream()
                response["status"] = "Stopped"
            elif command == "takeoff":
                success = controller.takeoff()
                response["status"] = "Taking off" if success else "Failed"
            elif command == "land":
                success = controller.land()
                response["status"] = "Landed" if success else "Failed"
            elif command == "battery":
                battery = controller.get_battery()
                response["battery"] = battery if battery is not None else "Unknown"
            elif command == "emergency":
                success = controller.emergency_stop()
                response["status"] = "Emergency stopped" if success else "Failed"
            else:
                response["error"] = "Unknown command"
            await websocket.send(json.dumps(response))
            if video_processor.streaming:
                ret, frame = video_processor.video_capture.read()
                if ret:
                    annotated_frame, detections = detector.detect_objects(frame)
                    success, track_id = tracker.track_object(detections, controller, frame)
                    if success and track_id:
                        for d in detections:
                            if d["label"].startswith("person"):
                                x1, y1, x2, y2 = d["bbox"]
                                cv2.putText(
                                    annotated_frame, f"ID: {track_id}",
                                    (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
                                )
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send(json.dumps({"frame": jpg_as_text, "detections": [d["label"] for d in detections], "track_id": track_id if success else None}))
                await asyncio.sleep(0.1)
    except Exception as e:
        print("WebSocket error:", str(e))

async def start_websocket_server(video_processor, detector, tracker, controller):
    async with websockets.serve(
        lambda ws: handle_tracking(ws, video_processor, detector, tracker, controller), "0.0.0.0", 8770
    ):
        await asyncio.Future()

def run_websocket(video_processor, detector, tracker, controller):
    asyncio.run(start_websocket_server(video_processor, detector, tracker, controller))

def main():
    controller = DroneController()
    
    if not controller.connect():
        print("Exiting due to connection failure")
        sys.exit(1)
    
    battery = controller.get_battery()
    if battery is None or battery < 20:
        print("Battery too low or unavailable, exiting")
        controller.disconnect()
        sys.exit(1)
    
    status_thread = threading.Thread(target=controller.monitor_status, args=(5,))
    status_thread.daemon = True
    status_thread.start()
    
    video_processor = VideoProcessor(controller)
    detector = ObjectDetector()
    tracker = DeepSortTracker()
    
    time.sleep(2)
    if not video_processor.start_stream():
        print("Video stream failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    websocket_thread = threading.Thread(target=run_websocket, args=(video_processor, detector, tracker, controller))
    websocket_thread.daemon = True
    websocket_thread.start()
    
    time.sleep(2)
    if not controller.takeoff():
        print("Takeoff failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    print("Tracking and streaming for 10 seconds. Press 'q' to stop early.")
    start_time = time.time()
    try:
        while time.time() - start_time < 10:
            if not video_processor.display_frame():
                print("Video display failed, landing drone")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("User interrupted, landing drone")
    
    time.sleep(1)
    controller.tello.send_rc_control(0, 0, 0, 0)
    if not controller.land():
        print("Landing failed, attempting emergency stop")
        controller.emergency_stop()
    
    video_processor.stop_stream()
    controller.get_battery()
    controller.disconnect()

if __name__ == "__main__":
    main()