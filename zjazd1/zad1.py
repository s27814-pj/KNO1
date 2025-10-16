FILENAME = "trained.keras"

import tensorflow as tf
import os
import argparse
import numpy as np


def main():
    if os.path.isfile(FILENAME):
        model = tf.keras.models.load_model(FILENAME)
    else:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist = model.fit(
            x_train, y_train, epochs=5
        )  # użyj verbose=0 jeśli jest problem z konsolą
        model.evaluate(x_test, y_test)

        model.save(FILENAME)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(hist.history["accuracy"])
        plt.title("Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")

        plt.subplot(1, 2, 2)
        plt.plot(hist.history["loss"])
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    parser = argparse.ArgumentParser(description="Number image to text number")
    parser.add_argument("--image_file", help="image file")
    args = parser.parse_args()
    if args.image_file:
        if os.path.isfile(args.image_file):
            img = tf.keras.utils.load_img(
                args.image_file, target_size=(28, 28), color_mode="grayscale"
            )
            input_arr = tf.keras.utils.img_to_array(img)
            input_arr = np.array([input_arr])

            prediction = model.predict(input_arr)
            digit = np.argmax(prediction)
            print(f"Predicted digit: {digit}")


if __name__ == "__main__":
    main()
