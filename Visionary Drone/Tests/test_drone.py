from drone_controller import DroneController
import time
import threading

def main():
    # Initialize drone controller
    controller = DroneController()
    
    # Connect to drone
    controller.connect()
    if not controller.connected:
        print("Exiting due to connection failure")
        return
    
    # Get initial battery level
    battery = controller.get_battery()
    if battery is None or battery < 20:
        print("Battery too low or unavailable, exiting")
        controller.disconnect()
        return
    
    ###(Only when it's more than 5 second)###
    # Start status monitoring in a separate thread
    # status_thread = threading.Thread(target=controller.monitor_status, args=(5,))
    # status_thread.daemon = True  # Stops thread when main program exits
    # status_thread.start()
    
    # Perform takeoff
    if controller.takeoff():
        print("Waiting 1 seconds in air")
        time.sleep(2)
    
    # Land the drone
    controller.land()
    
    # Get final battery level
    controller.get_battery()
    
    # Disconnect
    controller.disconnect()

if __name__ == "__main__":
    main()