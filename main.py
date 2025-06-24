from djitellopy import Tello
from time import sleep

drone=Tello()
drone.connect()

print(drone.get_battery())

# drone.takeoff()
# # drone.send_rc_control(0, 50, 0, 0)
# # sleep(2)
# # drone.send_rc_control(0, 0, 0, 0)
# # sleep(2)
# drone.send_rc_control(0, -50, 0, 0)
# sleep(1)
# drone.land()
