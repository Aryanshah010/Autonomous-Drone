import config
import asyncio
import websockets
import json
import threading
import sys

async def handle_config(websocket):
    """Handle WebSocket commands and send config status."""
    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command")
            if command == "check_config":
                config_status = {
                    "DRONE_IP": config.DRONE_IP,
                    "COMMAND_PORT": config.COMMAND_PORT,
                    "VIDEO_PORT": config.VIDEO_PORT,
                    "RESOLUTION": config.RESOLUTION,
                    "DEBUG": config.DEBUG,
                    "MODEL_PATH": config.MODEL_PATH,
                    "CONFIDENCE_THRESHOLD": config.CONFIDENCE_THRESHOLD,
                    "DEEP_SORT_MAX_AGE": config.DEEP_SORT_MAX_AGE,
                    "DEEP_SORT_IOU_THRESHOLD": config.DEEP_SORT_IOU_THRESHOLD,
                    "PID_GAINS": config.PID_GAINS,
                    "status": "Configuration loaded successfully"
                }
                await websocket.send(json.dumps(config_status))
            else:
                await websocket.send(json.dumps({"error": "Unknown command"}))
    except Exception as e:
        print("WebSocket error:", str(e))

async def start_websocket_server():
    """Start WebSocket server for config."""
    async with websockets.serve(handle_config, "0.0.0.0", 8765):
        await asyncio.Future()

def run_websocket():
    """Run WebSocket server in a thread."""
    asyncio.run(start_websocket_server())

def main():
    """Test configuration and start WebSocket server."""
    # Start WebSocket server
    websocket_thread = threading.Thread(target=run_websocket)
    websocket_thread.daemon = True
    websocket_thread.start()

    # Test configuration
    try:
        print("Testing configuration...")
        print(f"DRONE_IP: {config.DRONE_IP}")
        print(f"COMMAND_PORT: {config.COMMAND_PORT}")
        print(f"VIDEO_PORT: {config.VIDEO_PORT}")
        print(f"RESOLUTION: {config.RESOLUTION}")
        print(f"DEBUG: {config.DEBUG}")
        print(f"MODEL_PATH: {config.MODEL_PATH}")
        print(f"CONFIDENCE_THRESHOLD: {config.CONFIDENCE_THRESHOLD}")
        print(f"DEEP_SORT_MAX_AGE: {config.DEEP_SORT_MAX_AGE}")
        print(f"DEEP_SORT_IOU_THRESHOLD: {config.DEEP_SORT_IOU_THRESHOLD}")
        print(f"PID_GAINS: {config.PID_GAINS}")
        print("Configuration test passed!")
    except AttributeError as e:
        print(f"Configuration error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()