"""Step 4a: Local FastAPI server simulating a cloud inference endpoint."""
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI(title="Lab 1 Cloud Inference")
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print("TFLite model loaded; API ready on port 8000.")

class InferenceRequest(BaseModel):
    data: list[float] = Field(..., min_length=784, max_length=784)

@app.get("/")
def health():
    return {"status": "healthy"}

@app.post("/infer")
def infer(request: InferenceRequest):
    try:
        x = np.asarray(request.data, dtype=np.float32).reshape(1, 28, 28, 1)
        x = x.astype(inp["dtype"])
        start = time.perf_counter()
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        server_ms = (time.perf_counter() - start) * 1000
        scores = interpreter.get_tensor(out["index"])
        return {
            "prediction": int(np.argmax(scores)),
            "confidence": round(float(np.max(scores)), 4),
            "server_latency_ms": round(server_ms, 3),
            # Raw float32 array size; serialized JSON size is measured by client.
            "input_array_bytes": len(request.data) * 4,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
