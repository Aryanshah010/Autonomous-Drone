from drone_controller import DroneController
import time
import sys
import asyncio
import websockets
import json
import threading

async def handle_drone(websocket, controller):
    """Handle WebSocket commands and send drone status."""
    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command")
            response = {"command": command}
            if command == "connect":
                success = controller.connect()
                response["status"] = "Connected" if success else "Failed"
            elif command == "takeoff":
                success = controller.takeoff()
                response["status"] = "Taking off" if success else "Failed"
            elif command == "land":
                success = controller.land()
                response["status"] = "Landed" if success else "Failed"
            elif command == "battery":
                battery = controller.get_battery()
                response["battery"] = battery if battery is not None else "Unknown"
            elif command == "emergency":
                success = controller.emergency_stop()
                response["status"] = "Emergency stopped" if success else "Failed"
            else:
                response["error"] = "Unknown command"
            await websocket.send(json.dumps(response))
    except Exception as e:
        print("WebSocket error:", str(e))

async def start_websocket_server(controller):
    """Start WebSocket server for drone."""
    async with websockets.serve(lambda ws: handle_drone(ws, controller), "0.0.0.0", 8766):
        await asyncio.Future()

def run_websocket(controller):
    """Run WebSocket server in a thread."""
    asyncio.run(start_websocket_server(controller))

def main():
    controller = DroneController()
    
    # Start WebSocket server
    websocket_thread = threading.Thread(target=run_websocket, args=(controller,))
    websocket_thread.daemon = True
    websocket_thread.start()
    
    # Connect to drone
    if not controller.connect():
        print("Connection failed")
        sys.exit(1)
    
    # Get battery
    battery = controller.get_battery()
    if battery is None:
        print("Failed to get battery")
        controller.disconnect()
        sys.exit(1)
    print(f"Battery: {battery}%")
    
    # Takeoff
    time.sleep(2)
    if not controller.takeoff():
        print("Takeoff failed")
        controller.disconnect()
        sys.exit(1)
    
    # Hover for 5 seconds
    time.sleep(5)
    
    # Land
    if not controller.land():
        print("Landing failed, attempting emergency stop")
        controller.emergency_stop()
    
    # Get battery again
    battery = controller.get_battery()
    if battery is not None:
        print(f"Battery after landing: {battery}%")
    
    # Disconnect
    controller.disconnect()

if __name__ == "__main__":
    main()