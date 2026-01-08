from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Conv2D (batch_size, height, width, channels) aktualnie 32, 28, 28 - potrzeba 4D
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, y_train, x_test, y_test

def dense_model(x_train, y_train):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, x_train, y_train):
    trained_model = model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_split=0.2,
        verbose=1
    )
    return trained_model

def make_confusion_matrix(model, x_test, y_test, name):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Greens)
    plt.title("confusion matrix")
    plt.savefig(name)
    plt.close()


def cnn_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #okno 3x3 spwrawdza "kazde" 3x3 i daje wynik dla kazdego okna
        keras.layers.MaxPooling2D((2, 2)), #zmniejsza, okna 2x2 i max wartosc

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def cnn_model_aug():
    model = keras.Sequential([
        keras.layers.RandomFlip("horizontal"), #losowo lustro dla augmentacji
        keras.layers.RandomRotation(0.1), #aug
        keras.layers.RandomZoom(0.1),
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():

    x_train, y_train, x_test, y_test = load_data()
    # model1 = dense_model(x_train, y_train)
    # train_model(model1, x_train, y_train)
    # model1.save("dense.keras")
    model1 = keras.models.load_model("dense.keras")
    # make_confusion_matrix(model1, x_test, y_test, name="confusion_matrix_dense.png")
    loss = model1.evaluate(x_test, y_test, verbose=0)
    print("Loss, accuracy dla dense:", loss)

    # model2=cnn_model()
    # train_model(model2, x_train, y_train)
    # model2.save("cnn.keras")
    model2=keras.models.load_model("cnn.keras")
    # make_confusion_matrix(model2, x_test, y_test, name="confusion_matrix_cnn.png")
    loss2=model2.evaluate(x_test, y_test)
    print("Loss, accuracy dla cnn:", loss2)

    model3 = cnn_model_aug()
    train_model(model3, x_train, y_train)
    model3.save("cnn_aug.keras")
    model3=keras.models.load_model("cnn_aug.keras")
    # make_confusion_matrix(model3, x_test, y_test, name="confusion_matrix_cnn_aug.png")
    loss3=model3.evaluate(x_test, y_test)
    print("Loss, accuracy dla cnn_aug:", loss3)

if __name__ == "__main__":
    main()
