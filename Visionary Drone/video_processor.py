import cv2
import config
from drone_controller import DroneController
import time

class VideoProcessor:
    def __init__(self, drone_controller):
        """Initialize video processor with drone controller."""
        self.drone_controller = drone_controller
        self.video_capture = None
        self.streaming = False
        self.last_frame_time = 0
        self.frame_interval = 1 / 5  # Process at 15 FPS to reduce load

    def start_stream(self, retries=3, delay=2):
        """Start the drone's video stream with retries."""
        if not self.drone_controller.connected:
            print("Error: Drone not connected")
            return False
        
        for attempt in range(retries):
            try:
                # Send streamon command
                response = self.drone_controller.drone.send_command_with_return("streamon")
                if response.lower() != "ok":
                    print(f"Attempt {attempt + 1}: Failed to start video stream: {response}")
                    time.sleep(delay)
                    continue
                
                # Initialize OpenCV video capture
                video_url = f"udp://{config.DRONE_IP}:{config.VIDEO_PORT}?fifo_size=1000000&overrun_nonfatal=1"
                self.video_capture = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
                time.sleep(1)  # Wait for stream to stabilize
                if not self.video_capture.isOpened():
                    print(f"Attempt {attempt + 1}: Could not open video stream")
                    self.video_capture.release()
                    self.video_capture = None
                    time.sleep(delay)
                    continue
                
                self.streaming = True
                print("Video stream started successfully")
                return True
            except Exception as e:
                print(f"Attempt {attempt + 1}: Video stream start error: {str(e)}")
                time.sleep(delay)
        
        print("Failed to start video stream after retries")
        return False

    def stop_stream(self):
        """Stop the video stream and release resources."""
        if self.streaming:
            try:
                response = self.drone_controller.drone.send_command_with_return("streamoff")
                if response.lower() != "ok":
                    print("Failed to stop video stream:", response)
                self.streaming = False
            except Exception as e:
                print("Video stream stop error:", str(e))
        
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        cv2.destroyAllWindows()

    def display_frame(self):
        """Read and display a single frame, skipping if too frequent."""
        if not self.streaming or self.video_capture is None:
            print("Error: Video stream not active")
            return False
        
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return True  # Skip frame to maintain 15 FPS
        
        try:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Warning: Could not read frame, skipping")
                return True  # Skip corrupted frame
            
            # Resize frame to reduce processing load
            frame = cv2.resize(frame, (640, 360))  # Half resolution
            
            # Display frame if debug enabled
            if config.DEBUG:
                cv2.imshow("Tello Video", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Press 'q' to quit
                    return False
            self.last_frame_time = current_time
            return True
        except Exception as e:
            print("Frame display error:", str(e))
            return True  # Continue despite error