"""Deep Learning mini-project: CIFAR-10 image classification using a CNN."""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
tf.keras.utils.set_random_seed(SEED)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    os.makedirs("outputs", exist_ok=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train, y_test = y_train.squeeze(), y_test.squeeze()

    model = build_model()
    model.summary()
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=64,
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4%}")
    model.save("outputs/cifar10_cnn.keras")

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/accuracy_plot.png", dpi=150)
    plt.close()

    predictions = model.predict(x_test[:9], verbose=0).argmax(axis=1)
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i])
        true_name = CLASS_NAMES[int(y_test[i])]
        pred_name = CLASS_NAMES[int(predictions[i])]
        plt.title(f"True: {true_name}\nPred: {pred_name}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/sample_predictions.png", dpi=150)
    plt.close()
    print("Saved model and plots in the outputs/ folder.")

if __name__ == "__main__":
    main()
