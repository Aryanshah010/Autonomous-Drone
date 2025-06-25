from djitellopy import Tello
import time

# Create Tello object
tello = Tello()

# Connect to drone
tello.connect()
print("Battery:", tello.get_battery(), "%")

# Take off
tello.takeoff()
time.sleep(0.5)

# Rotate in place
# tello.rotate_clockwise(90)
# time.sleep(1)
# tello.rotate_counter_clockwise(90)

# Land
tello.land()
  