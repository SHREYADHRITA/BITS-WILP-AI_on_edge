"""Step 3: Benchmark local TFLite inference (simulated edge device)."""
import os
import time
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

if not os.path.exists("model.tflite"):
    raise FileNotFoundError("Run convert_to_tflite.py first.")

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print(f"Input shape: {inp['shape']}; dtype: {inp['dtype']}")
print(f"Output shape: {out['shape']}")

if os.path.exists("test_sample.npy"):
    x = np.load("test_sample.npy")
else:
    print("Warning: using a random test sample.")
    x = np.random.rand(1, 28, 28, 1).astype(np.float32)
x = x.astype(inp["dtype"])

# Warm-up, excluded from timing
interpreter.set_tensor(inp["index"], x)
interpreter.invoke()

times = []
for _ in range(100):
    start = time.perf_counter()
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    times.append((time.perf_counter() - start) * 1000)

times = np.asarray(times)
prediction = int(np.argmax(interpreter.get_tensor(out["index"])))
confidence = float(np.max(interpreter.get_tensor(out["index"])))
mean_ms = float(times.mean())

print("\n" + "=" * 48)
print("EDGE INFERENCE RESULTS")
print("=" * 48)
print(f"Predicted class : {prediction}")
print(f"Confidence      : {confidence:.4f}")
print(f"Mean latency    : {mean_ms:.3f} ms")
print(f"Median latency  : {np.median(times):.3f} ms")
print(f"p95 latency     : {np.percentile(times, 95):.3f} ms")
print(f"Max latency     : {times.max():.3f} ms")
print(f"Throughput      : {1000 / mean_ms:.1f} inferences/sec")
