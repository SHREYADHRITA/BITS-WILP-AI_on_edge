"""Step 4b: Measure cloud-simulation REST API round-trip time."""
import json
import os
import time
import numpy as np
import requests

BASE_URL = "http://127.0.0.1:8000"
URL = f"{BASE_URL}/infer"
N = 100
TIMEOUT = 10

if os.path.exists("test_sample.npy"):
    x = np.load("test_sample.npy")
else:
    print("Warning: using a random test sample.")
    x = np.random.rand(1, 28, 28, 1).astype(np.float32)

payload = {"data": x.astype(np.float32).flatten().tolist()}
payload_bytes = len(json.dumps(payload).encode("utf-8"))
print(f"JSON payload: {payload_bytes:,} bytes ({payload_bytes / 1024:.2f} KB)")

try:
    requests.get(BASE_URL, timeout=5).raise_for_status()
except requests.RequestException as exc:
    raise SystemExit("Start cloud_server.py in another terminal first.") from exc

requests.post(URL, json=payload, timeout=TIMEOUT).raise_for_status()  # warm-up
rtt_times, server_times = [], []
result = None
for _ in range(N):
    start = time.perf_counter()
    response = requests.post(URL, json=payload, timeout=TIMEOUT)
    elapsed = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    result = response.json()
    rtt_times.append(elapsed)
    server_times.append(result["server_latency_ms"])

rtt = np.asarray(rtt_times)
server = np.asarray(server_times)
mean_rtt = float(rtt.mean())
print("\n" + "=" * 48)
print("CLOUD INFERENCE RESULTS")
print("=" * 48)
print(f"Prediction          : {result['prediction']}")
print(f"Confidence          : {result['confidence']:.4f}")
print(f"Server mean latency : {server.mean():.3f} ms")
print(f"Server p95 latency  : {np.percentile(server, 95):.3f} ms")
print(f"RTT mean             : {mean_rtt:.3f} ms")
print(f"RTT p95              : {np.percentile(rtt, 95):.3f} ms")
print(f"Estimated overhead   : {(rtt - server).mean():.3f} ms")
print(f"Payload size         : {payload_bytes:,} bytes")
print(f"Bandwidth at 30 FPS  : {payload_bytes * 30 / 1024:.2f} KB/s")
