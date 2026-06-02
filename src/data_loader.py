import tensorflow as tf


def load_mnist_data():
    """
    Load and preprocess MNIST handwritten digit dataset.
    Returns train and test data.
    """

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from 0-255 to 0-1
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Flatten 28x28 images into 784 features
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    return X_train, y_train, X_test, y_test