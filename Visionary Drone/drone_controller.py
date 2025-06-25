from djitellopy import Tello
import config
import time

class DroneController:
    def __init__(self):
        """Initialize the Tello drone connection."""
        self.drone = Tello()
        self.connected = False
        self.flying = False

    def connect(self, retries=3, delay=2):
        """Connect to the drone with retries."""
        for attempt in range(retries):
            try:
                self.drone.connect()
                self.drone.set_speed(10)  # Slow speed for safety
                response = self.drone.send_command_with_return("command", timeout=5)
                if response.lower() == "ok":
                    self.connected = True
                    print("Drone connected and in SDK mode")
                    return True
                else:
                    print(f"Attempt {attempt + 1}: Failed to enter SDK mode: {response}")
                    time.sleep(delay)
            except Exception as e:
                print(f"Attempt {attempt + 1}: Connection error: {str(e)}")
                time.sleep(delay)
        print("Failed to connect after retries")
        return False

    def takeoff(self, retries=2, delay=1):
        """Command the drone to take off with retries."""
        if not self.connected:
            print("Error: Drone not connected")
            return False
        for attempt in range(retries):
            try:
                response = self.drone.send_command_with_return("takeoff", timeout=5)
                if response.lower() == "ok":
                    self.flying = True
                    print("Drone taking off")
                    return True
                else:
                    print(f"Attempt {attempt + 1}: Takeoff failed: {response}")
                    time.sleep(delay)
            except Exception as e:
                print(f"Attempt {attempt + 1}: Takeoff error: {str(e)}")
                time.sleep(delay)
        print("Takeoff failed after retries")
        return False

    def land(self, retries=2, delay=1):
        """Command the drone to land with retries."""
        if not self.connected:
            print("Error: Drone not connected")
            return False
        for attempt in range(retries):
            try:
                response = self.drone.send_command_with_return("land", timeout=5)
                if response.lower() == "ok":
                    self.flying = False
                    print("Drone landing")
                    return True
                else:
                    print(f"Attempt {attempt + 1}: Landing failed: {response}")
                    time.sleep(delay)
            except Exception as e:
                print(f"Attempt {attempt + 1}: Landing error: {str(e)}")
                time.sleep(delay)
        print("Landing failed after retries, attempting emergency stop")
        self.emergency_stop()
        return False

    def emergency_stop(self):
        """Stop all motors immediately."""
        try:
            response = self.drone.send_command_with_return("emergency", timeout=5)
            if response.lower() == "ok":
                self.flying = False
                self.connected = False
                print("Emergency stop activated")
                return True
            else:
                print("Emergency stop failed:", response)
                return False
        except Exception as e:
            print("Emergency stop error:", str(e))
            return False

    def get_battery(self):
        """Get the drone's battery level."""
        if not self.connected:
            print("Error: Drone not connected")
            return None
        try:
            battery = self.drone.get_battery()
            print("Battery level:", battery, "%")
            return battery
        except Exception as e:
            print("Battery query error:", str(e))
            return None

    def monitor_status(self, interval=5):
        """Monitor drone status periodically."""
        if not self.connected:
            print("Error: Drone not connected")
            return
        try:
            while self.connected:
                battery = self.get_battery()
                if battery is None:
                    print("Connection lost, attempting to land")
                    self.land()
                    break
                if battery < 15:
                    print("Low battery (<15%), landing drone")
                    self.land()
                    break
                if self.flying and not self.drone.get_current_state():
                    print("Flight state error, landing drone")
                    self.land()
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Status monitoring stopped by user")
            self.land()
        except Exception as e:
            print("Status monitoring error:", str(e))
            self.land()

    def disconnect(self):
        """End the drone connection safely."""
        if self.flying:
            print("Drone is flying, landing before disconnect")
            self.land()
        if self.connected:
            try:
                self.drone.end()
                self.connected = False
                print("Drone disconnected")
            except Exception as e:
                print("Disconnect error:", str(e))