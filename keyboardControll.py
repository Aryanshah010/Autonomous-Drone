from djitellopy import Tello
from pynput import keyboard
from time import sleep

drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

SPEED = 50  # cm/s

def on_press(key):
    try:
        if key.char == 't':
            drone.takeoff()
        elif key.char == 'l':
            drone.land()
        elif key.char == 'w':
            
            drone.send_rc_control(0, 0, SPEED, 0)  # Up
        elif key.char == 's':
            drone.send_rc_control(0, 0, -SPEED, 0)  # Down
        elif key.char == 'a':
            drone.send_rc_control(0, 0, 0, -SPEED)  # Rotate left
        elif key.char == 'd':
            drone.send_rc_control(0, 0, 0, SPEED)  # Rotate right
        elif key.char == 'q':
            print("Quitting...")
            drone.land()
            return False  # Stop listener
    except AttributeError:
        # Special keys (arrows)
        if key == keyboard.Key.up:
            drone.send_rc_control(0, SPEED, 0, 0)  # Forward
        elif key == keyboard.Key.down:
            drone.send_rc_control(0, -SPEED, 0, 0)  # Backward
        elif key == keyboard.Key.left:
            drone.send_rc_control(-SPEED, 0, 0, 0)  # Left
        elif key == keyboard.Key.right:
            drone.send_rc_control(SPEED, 0, 0, 0)  # Right

def on_release(key):
    # Stop movement when key is released
    drone.send_rc_control(0, 0, 0, 0)

# Start listening to keyboard
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

  

