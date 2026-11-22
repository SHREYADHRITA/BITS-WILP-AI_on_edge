BITS Pilani Lab Experiment 1 — Foundations of Edge AI

Files:
  build_model.py       Step 1: train MNIST CNN
  convert_to_tflite.py Step 2: dynamic-range TFLite conversion
  local_inference.py   Step 3: local edge benchmark
  cloud_server.py      Step 4a: FastAPI inference server
  cloud_client.py      Step 4b: cloud API benchmark
  compare_results.py  Step 5: combined comparison

Recommended environment: Python 3.9–3.11.
Install:
  pip install --upgrade pip
  pip install "tensorflow>=2.13,<2.17" fastapi uvicorn requests numpy

Run in order:
  python build_model.py
  python convert_to_tflite.py
  python local_inference.py

Terminal 1:
  python cloud_server.py

Terminal 2:
  python cloud_client.py
  python compare_results.py

Keep all scripts in the same directory.

Notes:
- Code follows the handout's specified CNN architecture and workflow.
- The stated architecture (28x28 input, Conv2D(8), 2x2 pooling, Flatten,
  Dense(32), Dense(10)) has approximately 50,618 trainable parameters,
  not approximately 5,900 as stated in the handout. The architecture itself
  is preserved as specified.
- Localhost FastAPI timings are not representative of a real remote cloud/WAN.
- JSON payload size is measured from the serialized request on the client.
