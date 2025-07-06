from djitellopy import Tello

drone = Tello()
drone.connect()
drone.get_battery()
drone.takeoff()
drone.send_rc_control(0, 0, 0, 0)
drone.land()