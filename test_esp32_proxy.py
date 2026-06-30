import os, requests

print("=== Proxy env vars ===")
for k in ["HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","ALL_PROXY","NO_PROXY"]:
    print(f"  {k} = {os.environ.get(k, '(not set)')}")

print()
print("=== Test WITHOUT proxy bypass ===")
try:
    r = requests.get("http://10.248.116.230/sensor", timeout=2)
    print("  OK:", r.json())
except Exception as e:
    print("  FAIL:", e)

print()
print("=== Test WITH proxy bypass ===")
try:
    r = requests.get("http://10.248.116.230/sensor", timeout=2, proxies={"http": None, "https": None})
    print("  OK:", r.json())
except Exception as e:
    print("  FAIL:", e)

print()
print("=== Test using urllib (no proxy at all) ===")
try:
    import urllib.request
    req = urllib.request.Request("http://10.248.116.230/sensor")
    with urllib.request.urlopen(req, timeout=2) as resp:
        import json
        data = json.loads(resp.read().decode())
        print("  OK:", data)
except Exception as e:
    print("  FAIL:", e)
