import argparse
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps

#python .\use_model.py shirt.jpg
#python .\use_model.py bag.jpg


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def prep_image(image):
    img = Image.open(image)

    # skala szarości
    img = img.convert("L")

    # zmiana rozmiaru
    img = img.resize((28, 28))

    # negatyw
    img = ImageOps.invert(img)

    # zamiana na numpy
    img_array = np.array(img)

    # normalizacja
    img_array = img_array / 255.0

    # dodanie wymiarów: (1, 28, 28, 1)
    img_array = img_array[np.newaxis, ..., np.newaxis]

    return img_array

def predict(model, image):
    predictions = model.predict(image)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions) #index of the largest value in the array

    return predicted_class, confidence, predictions

def main():
    parser = argparse.ArgumentParser(description="image")
    parser.add_argument("image", help="image file")
    args = parser.parse_args()

    model_dense = keras.models.load_model("dense.keras")
    model_cnn = keras.models.load_model("cnn.keras")
    model_cnn_aug = keras.models.load_model("cnn_aug.keras")

    image= prep_image(args.image)

    class_id, confidence, list_confidence = predict(model_dense, image)

    print("dense Predicted:", CLASS_NAMES[class_id])
    print(f"dense's Confidence: {confidence:.4f}")
    print(list_confidence)

    class_id, confidence, list_confidence = predict(model_cnn, image)

    print("cnn Predicted:", CLASS_NAMES[class_id])
    print(f"cnn's Confidence: {confidence:.4f}")
    print(list_confidence)

    class_id, confidence, list_confidence = predict(model_cnn_aug, image)

    print("cnn_aug Predicted:", CLASS_NAMES[class_id])
    print(f"cnn_aug's Confidence: {confidence:.4f}")
    print(list_confidence)


if __name__ == "__main__":
    main()
