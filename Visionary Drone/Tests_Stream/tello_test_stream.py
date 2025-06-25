from djitellopy import Tello
import time
import cv2
import base64
import asyncio
import websockets
import threading
import numpy as np

async def send_video_frames(websocket, tello, cap):
    """Send video frames to WebSocket clients."""
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame")
                await asyncio.sleep(0.1)
                continue
            # Encode frame as JPEG with 100% quality
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(jpg_as_text)
            await asyncio.sleep(0.1)  # ~10 FPS
    except Exception as e:
        print("WebSocket error:", str(e))

async def start_websocket_server(tello, cap):
    """Start WebSocket server for video."""
    async with websockets.serve(
        lambda ws: send_video_frames(ws, tello, cap), "0.0.0.0", 8769
    ):
        await asyncio.Future()

def run_websocket(tello, cap):
    """Run WebSocket server in a thread."""
    asyncio.run(start_websocket_server(tello, cap))

def main():
    # Create Tello object
    tello = Tello()

    # Connect to drone
    try:
        tello.connect()
        battery = tello.get_battery()
        print("Battery:", battery, "%")
        if battery < 20:
            print("Battery too low, exiting")
            tello.end()
            return
    except Exception as e:
        print("Connection failed:", str(e))
        return

    # Start video stream
    try:
        tello.streamon()
        time.sleep(2)  # Wait for stream stability
        cap = cv2.VideoCapture('udp://192.168.10.1:11111')
        if not cap.isOpened():
            print("Failed to open video stream")
            tello.streamoff()
            tello.end()
            return
    except Exception as e:
        print("Video stream error:", str(e))
        tello.end()
        return

    # Start WebSocket server
    websocket_thread = threading.Thread(target=run_websocket, args=(tello, cap))
    websocket_thread.daemon = True
    websocket_thread.start()

    # Take off
    try:
        tello.takeoff()
        time.sleep(0.5)
    except Exception as e:
        print("Takeoff failed:", str(e))
        tello.land()
        tello.streamoff()
        cap.release()
        tello.end()
        return

    # Land
    try:
        tello.land()
    except Exception as e:
        print("Landing failed:", str(e))
        tello.emergency()  # Emergency stop if landing fails
    finally:
        tello.streamoff()
        cap.release()
        tello.end()

if __name__ == "__main__":
    main()