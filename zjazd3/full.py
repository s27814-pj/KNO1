import argparse
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np

# python full.py --alcohol 12.2 --malic 1.1 --ash 2.2 --alcalinity 16 --magnesium 101 --total_phenols 2.05 --flavanoids 1.09 --nonflavanoid 0.63 --proanthocyanins 1.0 --color 3.27 --hue 1.05 --od 3.5 --proline 1080

def load_data():
    wine_data=pd.read_csv(
        "wine.csv",
        names=["Class","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
               "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins","Color intensity",
               "Hue", "OD280/OD315 of diluted wines", "Proline"
               ])

    one_hot = pd.get_dummies(wine_data, columns=["Class"])

    features = one_hot.drop(columns=["Class_1", "Class_2", "Class_3"])
    labels = one_hot[["Class_1", "Class_2", "Class_3"]] #jedynkowa macierz dla labeli
    features, labels = shuffle(features, labels, random_state=42)

    scaler = StandardScaler()
    features = scaler.fit_transform(features) #bez standardowania accuracy niskie okolo 35%


    return features, labels

def small_model(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels.values, test_size=0.2, random_state=3)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model_fit = model.fit(
        x_train, y_train,
        epochs=25,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    model.save("model1.keras", overwrite=True)


def big_model(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels.values, test_size=0.2, random_state=3)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='sigmoid', name="nazwa"),
        # tf.keras.layers.Dense(32, activation='softmax', name="nazwaKolejnej"),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model_fit = model.fit(
        x_train, y_train,
        epochs=25,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    model.save("model2.keras", overwrite=True)

def plot(model_fit1,model_fit2,):
    plt.figure(figsize=(10, 4))

    plt.plot(model_fit1.history["accuracy"], label="Model 1")
    plt.plot(model_fit2.history["accuracy"], label="Model 2")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def main():

    features, labels = load_data()

    print(features)
    print(labels)
    # small_model(features, labels)
    # big_model(features, labels)
    model1 = tf.keras.models.load_model("model1.keras")
    model2 = tf.keras.models.load_model("model2.keras")
    # plot(model1.fit(features, labels,epochs=25),model2.fit(features, labels,epochs=25))

    parser = argparse.ArgumentParser(
        description="Klasyfikacja dla podanych"
    )

    parser.add_argument("--alcohol", type=float, required=True)
    parser.add_argument("--malic", type=float, required=True)
    parser.add_argument("--ash", type=float, required=True)
    parser.add_argument("--alcalinity", type=float, required=True)
    parser.add_argument("--magnesium", type=float, required=True)
    parser.add_argument("--total_phenols", type=float, required=True)
    parser.add_argument("--flavanoids", type=float, required=True)
    parser.add_argument("--nonflavanoid", type=float, required=True)
    parser.add_argument("--proanthocyanins", type=float, required=True)
    parser.add_argument("--color", type=float, required=True)
    parser.add_argument("--hue", type=float, required=True)
    parser.add_argument("--od", type=float, required=True)
    parser.add_argument("--proline", type=float, required=True)
    args = parser.parse_args()


    input_data = np.array(
        [
            [
                args.alcohol,
                args.malic,
                args.ash,
                args.alcalinity,
                args.magnesium,
                args.total_phenols,
                args.flavanoids,
                args.nonflavanoid,
                args.proanthocyanins,
                args.color,
                args.hue,
                args.od,
                args.proline,
            ]
        ]


    )

    prediction = model2.predict(input_data)
    predicted_class = np.argmax(prediction[0]) + 1  #plus jeden bo 123 nie 012
    print(prediction)
    print(f"Oczekiwana kategoria wina: {predicted_class}")

if __name__ == "__main__":
    main()