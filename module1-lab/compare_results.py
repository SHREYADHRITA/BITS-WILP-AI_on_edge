"""Step 5: Side-by-side edge vs cloud REST RTT comparison."""
import json
import os
import time
import numpy as np
import requests

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL = "model.tflite"
BASE_URL = "http://127.0.0.1:8000"
URL = f"{BASE_URL}/infer"
N = 100
TIMEOUT = 10

if not os.path.exists(MODEL):
    raise FileNotFoundError("Run convert_to_tflite.py first.")
if os.path.exists("test_sample.npy"):
    x = np.load("test_sample.npy")
else:
    print("Warning: using a random test sample.")
    x = np.random.rand(1, 28, 28, 1).astype(np.float32)

# Edge benchmark
interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
edge_x = x.astype(inp["dtype"])
interpreter.set_tensor(inp["index"], edge_x)
interpreter.invoke()  # warm-up

edge_times = []
for _ in range(N):
    start = time.perf_counter()
    interpreter.set_tensor(inp["index"], edge_x)
    interpreter.invoke()
    edge_times.append((time.perf_counter() - start) * 1000)
edge_prediction = int(np.argmax(interpreter.get_tensor(out["index"])))

# Cloud benchmark
payload = {"data": x.astype(np.float32).flatten().tolist()}
payload_bytes = len(json.dumps(payload).encode("utf-8"))
try:
    requests.get(BASE_URL, timeout=5).raise_for_status()
except requests.RequestException as exc:
    raise SystemExit("Start cloud_server.py in another terminal first.") from exc

requests.post(URL, json=payload, timeout=TIMEOUT).raise_for_status()  # warm-up
cloud_times = []
cloud_prediction = None
for _ in range(N):
    start = time.perf_counter()
    response = requests.post(URL, json=payload, timeout=TIMEOUT)
    elapsed = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    cloud_times.append(elapsed)
    cloud_prediction = int(response.json()["prediction"])

edge = np.asarray(edge_times)
cloud = np.asarray(cloud_times)
edge_mean, cloud_mean = float(edge.mean()), float(cloud.mean())

print("\n" + "=" * 68)
print(f'{"METRIC":<28} {"EDGE":>18} {"CLOUD RTT":>18}')
print("=" * 68)
print(f'{"Mean latency (ms)":<28} {edge_mean:>18.3f} {cloud_mean:>18.3f}')
print(f'{"Median latency (ms)":<28} {np.median(edge):>18.3f} {np.median(cloud):>18.3f}')
print(f'{"p95 latency (ms)":<28} {np.percentile(edge, 95):>18.3f} {np.percentile(cloud, 95):>18.3f}')
print(f'{"Throughput (inf/sec)":<28} {1000 / edge_mean:>18.1f} {1000 / cloud_mean:>18.1f}')
print(f'{"Data sent per call":<28} {"0 bytes":>18} {f"{payload_bytes:,} B":>18}')
print(f'{"Network required":<28} {"No":>18} {"Yes":>18}')
print(f'{"Prediction":<28} {edge_prediction:>18} {cloud_prediction:>18}')
print("=" * 68)
print(f"Edge speedup (cloud RTT / edge): {cloud_mean / edge_mean:.2f}x")
print(f"Bandwidth per inference       : {payload_bytes:,} bytes ({payload_bytes / 1024:.2f} KB)")
print(f"Bandwidth at 30 FPS           : {payload_bytes * 30 / 1024:.2f} KB/s")
print(f"Predictions match              : {edge_prediction == cloud_prediction}")
