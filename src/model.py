import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def build_model(input_shape=784, learning_rate=0.001):
    """
    Build a neural network for handwritten digit recognition.
    """

    model = Sequential(
        [
            tf.keras.Input(shape=(input_shape,)),
            Dense(128, activation="relu", name="hidden_layer_1"),
            Dropout(0.2),
            Dense(64, activation="relu", name="hidden_layer_2"),
            Dense(10, activation="linear", name="output_layer"),
        ],
        name="digit_recognition_model",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model