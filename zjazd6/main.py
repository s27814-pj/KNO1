import tensorflow as tf
from tensorflow.keras import layers, losses, Model
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
DATA_DIR = "photos"  # folder with at least 20 images

# === DATA AUGMENTATION ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# === LOAD DATASET ===
def load_dataset():
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        labels=None,          # brak etykiet
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    # normalizacja 0–255 → 0–1
    dataset = dataset.map(lambda x: x / 255.0)
    return dataset

# === AUTOENCODER MODEL ===
class Autoencoder(Model):
    def __init__(self, latent_dim=2):
        super().__init__()
        # ENCODER
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Flatten(),
            layers.Dense(latent_dim, activation="relu")
        ])
        # DECODER
        self.decoder = tf.keras.Sequential([
            layers.Dense(128 * 128 * 3, activation="sigmoid"),
            layers.Reshape((128, 128, 3))
        ])

    def call(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# === PREPARE TRAINING DATASET ===
def prepare_training_dataset(dataset):
    augmented_images = []
    original_images = []
    for batch in dataset:
        augmented = data_augmentation(batch)
        augmented_images.append(augmented)
        original_images.append(batch)
    x_train = tf.concat(original_images, axis=0)
    y_train = tf.concat(augmented_images, axis=0)
    return tf.data.Dataset.from_tensor_slices((y_train, x_train)).batch(BATCH_SIZE)

# === SHOW RESULTS ===
def show_results(model, dataset, n=5):
    for batch in dataset.take(1):
        originals, _ = batch
        reconstructions = model(originals)

    plt.figure(figsize=(15, 4))
    for i in range(n):
        # original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(originals[i].numpy())
        plt.title("Original")
        plt.axis("off")

        # reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].numpy())
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()

# === MAIN ===
def main():
    dataset = load_dataset()
    train_ds = prepare_training_dataset(dataset)

    autoencoder = Autoencoder(latent_dim=2)
    autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

    autoencoder.fit(train_ds, epochs=20)

    show_results(autoencoder, train_ds)

if __name__ == "__main__":
    main()
