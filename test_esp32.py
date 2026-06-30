import sys
import requests
import time

def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.100"
    url = f"http://{ip}/status"
    print(f"Connecting to ESP32 server at {url}...")
    print("Press Ctrl+C to stop.")
    
    while True:
        try:
            resp = requests.get(url, timeout=1.0)
            if resp.status_code == 200:
                print(f"[{time.strftime('%H:%M:%S')}] ESP32 Data: {resp.text.strip()}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] HTTP Error: {resp.status_code}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Connection Error: {e}")
        time.sleep(1.0)

if __name__ == "__main__":
    main()
