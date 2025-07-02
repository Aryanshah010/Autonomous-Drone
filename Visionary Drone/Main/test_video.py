from drone_controller import DroneController
from video_processor import VideoProcessor
import time
import threading
import sys

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
    
    # Initialize video processor
    video_processor = VideoProcessor(controller)
    
    # Start video stream
    time.sleep(2)  # Delay for connection stability
    if not video_processor.start_stream():
        print("Video stream failed, landing and exiting")
        controller.land()
        controller.disconnect()
        sys.exit(1)
    
    # Perform takeoff
    # time.sleep(2)  # Delay for stream stability
    # if not controller.takeoff():
    #     print("Takeoff failed, landing and exiting")
    #     controller.land()
    #     controller.disconnect()
    #     sys.exit(1)
    
    # Display video for 5 seconds
    print("Displaying video for 5 seconds. Press 'q' to stop early.")
    start_time = time.time()
    try:
        while time.time() - start_time < 30:
            if not video_processor.display_frame():
                print("Video display failed, landing drone")
                break
            time.sleep(0.05)  # Small delay to reduce CPU load
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