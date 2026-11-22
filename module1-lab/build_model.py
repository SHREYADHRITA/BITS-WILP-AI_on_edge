"""Step 1: Train a compact CNN on MNIST."""
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(f"Training samples: {len(x_train):,}")
print(f"Test samples: {len(x_test):,}")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
], name="mnist_cnn")

model.summary()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64,
          validation_split=0.1, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

model.save("mnist_model.keras")
np.save("test_sample.npy", x_test[:1])
np.save("test_label.npy", y_test[:1])
print("Saved mnist_model.keras, test_sample.npy and test_label.npy")
