import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
IMG_CHANNELS = 3
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    
    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_image(path):
    """
    Load image from path `path`.

    Return an image cropped and scaled to IMG_WIDTH x IMG_HEIGHT, formatted as
    a IMG_WIDTH x IMG_HEIGHT x IMG_CHANNELS numpy ndarray.
    """
    i = cv2.imread(path, cv2.IMREAD_COLOR if IMG_CHANNELS == 3 else cv2.IMREAD_GRAYSCALE)
    if i is not None:
        if IMG_CHANNELS == 3:
            h, w, _ = i.shape
        else:
            h, w = i.shape
        # aspect ratio
        if w * IMG_HEIGHT > h * IMG_WIDTH:
            cw = (h * IMG_WIDTH) // IMG_HEIGHT
            cx = (w - cw) // 2
            i = i[:, cx:(cx + cw)]
            w = cw
        elif w * IMG_HEIGHT < h * IMG_WIDTH:
            ch = (w * IMG_HEIGHT) // IMG_WIDTH
            cy = (h - ch) // 2
            i = i[cy:(cy + ch), :]
            h = ch
        # size
        a = w * h
        ta = IMG_WIDTH * IMG_HEIGHT
        if a != ta:
            i = cv2.resize(i, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_AREA if a > ta else cv2.INTER_CUBIC)
        i = np.reshape(i, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    return i


def load_data(data_dir, verbose=0):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x IMG_CHANNELS.
    `labels` should be a list of integer labels, representing the categories
    for each of the corresponding `images`.
    """
    files = []
    for c in range(NUM_CATEGORIES):
        d = os.path.join(data_dir, str(c))
        if os.path.isdir(d):
            with os.scandir(d) as sd:
                for e in sd:
                    if e.is_file:
                        files.append((c, e.path))
    if verbose > 0:
        print(f"Load images ({len(files)})")
    images, labels = [], []
    for (i, (c, p)) in enumerate(files):
        if verbose > 0:
            fs = len(files)
            e = '' if (i + 1) < fs else '\n'
            print(f"\r{i + 1}/{fs} [{'=' * int((i / fs) * 25) + ('>' if (i + 1) < fs else '='):.<25}]", end=e)
        i = load_image(p)
        if i is not None:
            images.append(i)
            labels.append(c)
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
        tf.keras.layers.Lambda(lambda x: x / 255.0),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
