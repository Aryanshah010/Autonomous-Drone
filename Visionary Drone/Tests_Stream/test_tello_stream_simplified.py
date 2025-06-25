from djitellopy import Tello
import cv2
import base64
import asyncio
import websockets
import threading
import time

async def send_video(websocket, cap):
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                await websocket.send(jpg_as_text)
            await asyncio.sleep(0.1)  # ~10 FPS
    except:
        pass

async def start_websocket(cap):
    async with websockets.serve(lambda ws: send_video(ws, cap), "0.0.0.0", 8769):
        await asyncio.Future()

def run_websocket(cap):
    asyncio.run(start_websocket(cap))

def main():
    tello = Tello()
    try:
        tello.connect()
        print("Battery:", tello.get_battery(), "%")
        tello.streamon()
        cap = cv2.VideoCapture('udp://192.168.10.1:11111')
        if not cap.isOpened():
            print("Video failed")
            tello.streamoff()
            tello.end()
            return
        websocket_thread = threading.Thread(target=run_websocket, args=(cap,))
        websocket_thread.daemon = True
        websocket_thread.start()
        tello.takeoff()
        time.sleep(0.5)
        tello.land()
    except:
        tello.land()
        tello.emergency()
    finally:
        tello.streamoff()
        cap.release()
        tello.end()

if __name__ == "__main__":
    main()