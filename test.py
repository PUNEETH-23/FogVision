import requests
import time
ip='10.19.13.230'
while True:
    data=requests.get(f"http://{ip}/sensor").json()
    print(data)
    time.sleep(0.5)
