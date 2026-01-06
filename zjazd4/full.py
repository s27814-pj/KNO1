import argparse
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import keras_tuner as kt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# python full.py --alcohol 12.2 --malic 1.1 --ash 2.2 --alcalinity 16 --magnesium 101 --total_phenols 2.05 --flavanoids 1.09 --nonflavanoid 0.63 --proanthocyanins 1.0 --color 3.27 --hue 1.05 --od 3.5 --proline 1080

# base line accuracy: 0.8496 - loss: 0.5266 - val_accuracy: 0.8276 - val_loss: 0.5461 epoch: 25
# nowy model accuracy: 0.9823 - loss: 0.2373 - val_accuracy: 1.0000 - val_loss: 0.2055 epoch 4
def load_data():
    wine_data = pd.read_csv(
        "wine.csv",
        names=["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
               "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity",
               "Hue", "OD280/OD315 of diluted wines", "Proline"
               ])

    one_hot = pd.get_dummies(wine_data, columns=["Class"])

    features = one_hot.drop(columns=["Class_1", "Class_2", "Class_3"])
    labels = one_hot[["Class_1", "Class_2", "Class_3"]]  # jedynkowa macierz dla labeli
    features, labels = shuffle(features, labels, random_state=42)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # bez standardowania accuracy niskie okolo 35%

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


def plot(model_fit1, model_fit2, ):
    plt.figure(figsize=(10, 4))

    plt.plot(model_fit1.history["accuracy"], label="Model 1")
    plt.plot(model_fit2.history["accuracy"], label="Model 2")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, x, y, title):
    y_pred = model.predict(x)

    y_true = np.argmax(y, axis=1) #index najwiekszej wartosci dla labeli bo one-hot
    y_pred = np.argmax(y_pred, axis=1) #index najwiekszej prawdopodobientswa przynalezenia

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[1, 2, 3]
    )
    disp.plot(cmap=plt.cm.Greens)
    plt.title(title)
    plt.show()


def model_builder(hp):
    model = keras.Sequential()
    hp_units = hp.Int('units', min_value=8, max_value=128, step=8)
    hp_activation = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh'])
    model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))
    model.add(keras.layers.Dense(3, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),  # one-hot labels
                  metrics=['accuracy'])
    model.save("model3tuned.keras", overwrite=True)
    return model


def main():
    features, labels = load_data()

    print(features)
    print(labels)
    x_train, x_test, y_train, y_test = train_test_split(features, labels.values, test_size=0.2, random_state=3)

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt_full_maybe')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(x_train, y_train, epochs=50, validation_split=0.2,
                 batch_size=kt.engine.hyperparameters.HyperParameters().Choice('batch_size', [8, 16, 32, 64]),
                 callbacks=[stop_early])
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}. and activation function is {best_hps.get('activation')}.
    """)
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train, y_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

    eval_result = hypermodel.evaluate(x_test, y_test)
    print("[test loss, test accuracy]:", eval_result)

    # small_model(features, labels)
    big_model(features, labels)
    # model1 = tf.keras.models.load_model("model1.keras")
    model2 = tf.keras.models.load_model("model2.keras")
    plot(model2.fit(x_train, y_train, epochs=25),
         hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2))

    plot_confusion_matrix(
        model2,
        x_test,
        y_test,
        title="Confusion Matrix Big Model"
    )

    plot_confusion_matrix(
        hypermodel,
        x_test,
        y_test,
        title="Confusion Matrix Tuned Model"
    )

    tuner.results_summary()
    print('---')
    hypermodel.summary()

if __name__ == "__main__":
    main()
