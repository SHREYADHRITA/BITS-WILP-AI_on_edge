"""Step 2: Convert the Keras model to TFLite with dynamic-range quantization."""
import os
import tensorflow as tf

source = "mnist_model.keras"
target = "model.tflite"
if not os.path.exists(source):
    raise FileNotFoundError("Run build_model.py first; mnist_model.keras is missing.")

model = tf.keras.models.load_model(source)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(target, "wb") as f:
    f.write(tflite_model)

keras_kb = os.path.getsize(source) / 1024
tflite_kb = os.path.getsize(target) / 1024
print("Conversion complete!")
print(f"Keras model size : {keras_kb:,.2f} KB")
print(f"TFLite model size: {tflite_kb:,.2f} KB")
print(f"Compression ratio: {keras_kb / tflite_kb:.2f}x")
