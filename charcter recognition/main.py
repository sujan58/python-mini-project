# Small MNIST experiment – just trying out a simple MLP.
# (Nothing fancy, just something I can tweak later.)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Just to keep results mostly consistent
np.random.seed(42)
tf.random.set_seed(42)


def load_data_and_summary():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Quick check on label distribution
    labels, counts = np.unique(y_train, return_counts=True)
    print("Label counts (train):")
    for l, c in zip(labels, counts):
        print(f"{l}: {c}")

    return (x_train, y_train), (x_test, y_test)


def show_some_images(x, y, count=40):
    # Show a random set of digits just to get a feel for the dataset
    idx = np.random.choice(len(x), size=count, replace=False)
    chosen_x = x[idx]
    chosen_y = y[idx]

    rows = count // 10 if count % 10 == 0 else (count // 10) + 1
    plt.figure(figsize=(10, rows * 1.2))

    for i in range(count):
        plt.subplot(rows, 10, i + 1)
        plt.imshow(chosen_x[i], cmap="gray")
        plt.axis("off")
        plt.title(str(chosen_y[i]), fontsize=8)

    plt.tight_layout()
    plt.show()


def preprocess(x_train, x_test, y_train, y_test):
    # normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # one-hot
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test


def get_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_training(hist):
    # accuracy
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], "--", label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], "--", label="val")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def show_predictions(model, x_test, y_test, n=10):
    # show the first n predictions
    preds = model.predict(x_test[:n])
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(10, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(x_test[i], cmap="gray")
        plt.axis("off")
        t = np.argmax(y_test[i])
        p = pred_labels[i]
        plt.title(f"P:{p} / T:{t}", fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    (x_train, y_train), (x_test, y_test) = load_data_and_summary()

    print("\nShowing some random training images...")
    show_some_images(x_train, y_train, count=40)

    print("\nPreprocessing data...")
    x_train_p, x_test_p, y_train_p, y_test_p = preprocess(
        x_train, x_test, y_train, y_test
    )

    print("\nBuilding model...")
    model = get_model()
    model.summary()

    print("\nTraining...\n")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    hist = model.fit(
        x_train_p,
        y_train_p,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nTraining complete. Plotting history...")
    plot_training(hist)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test_p, y_test_p, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    print("\nSample predictions:")
    show_predictions(model, x_test_p, y_test_p, n=10)

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/mnist_mlp.h5")
    print("\nModel saved to saved_models/mnist_mlp.h5")


if __name__ == "__main__":
    main()
