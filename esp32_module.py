import requests
import time
from typing import Dict, Any, Optional

def get_sensor_data(
    ip: str, timeout: float = 0.1) -> Optional[Dict[str, float]]:
    """
    Fetch VL53L0X distance sensors from the ESP32 car.
    
    Endpoint: GET http://{ip}/sensor
    Expected Response: {"left": int, "center": int, "right": int} (in mm)
    
    Returns:
        Dict mapped to meters: {"left": float, "middle": float, "right": float} or None if failed.
    """
    url = f"http://{ip}/sensor"
    try:
        resp = requests.get(url, timeout=timeout, proxies={"http": None, "https": None})
        if resp.status_code == 200:
            data = resp.json()
            # Convert mm to meters
            return {
                "left": float(data.get("left", 8000.0)) / 1000.0,
                "center": float(data.get("center", 8000.0)) / 1000.0,
                "right": float(data.get("right", 8000.0)) / 1000.0
            }
    except Exception:
        pass
    return None

def send_motor_speed(ip: str, speed: int, turn: int = 0, timeout: float = 0.1) -> bool:
    """
    Send target motor speed and turn rate to the ESP32 car.
    
    Endpoint: POST http://{ip}/motor
    Payload: {"speed": int, "turn": int} (Range: -100 to 100)
    
    Returns:
        True if sent successfully, False otherwise.
    """
    url = f"http://{ip}/motor"
    try:
        resp = requests.post(url, json={"speed": speed, "turn": turn}, timeout=timeout, proxies={"http": None, "https": None})
        return resp.status_code == 200
    except Exception:
        pass
    return False

if __name__ == "__main__":
    import keyboard
    import sys
    import os
    
    ESP32_IP = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ESP32_IP", "localhost")
    print(f"=== ESP32 Car Standalone keyboard Controller (IP: {ESP32_IP}) ===")
    print("UP    = Forward")
    print("DOWN  = Reverse")
    print("LEFT  = Turn Left")
    print("RIGHT = Turn Right")
    print("SPACE = Stop")
    print("ESC   = Exit\n")
    
    # Ensure stopped at startup
    send_motor_speed(ESP32_IP, 0, 0)
    
    while True:
        speed = 0
        turn = 0
        
        # Movement
        if keyboard.is_pressed("up"):
            speed = 100
        elif keyboard.is_pressed("down"):
            speed = -100
            
        # Steering
        if keyboard.is_pressed("left"):
            turn = -100
        elif keyboard.is_pressed("right"):
            turn = 100
            
        # Emergency stop
        if keyboard.is_pressed("space"):
            speed = 0
            turn = 0
            
        send_motor_speed(ESP32_IP, speed, turn)
        
        # Exit
        if keyboard.is_pressed("esc"):
            send_motor_speed(ESP32_IP, 0, 0)
            print("\nStopping car...")
            break
            
        time.sleep(0.05)
    
    print("Program terminated.")