import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_image(image_path):
    """
    Convert input image into MNIST-like format.
    """

    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0

    # Invert image if needed: white background, black digit
    image_array = 1.0 - image_array

    image_array = image_array.reshape(1, 28 * 28)

    return image_array


def predict_digit(image_path):
    model = tf.keras.models.load_model("models/digit_model.keras")

    processed_image = preprocess_image(image_path)

    logits = model.predict(processed_image)
    probabilities = tf.nn.softmax(logits).numpy()[0]

    predicted_digit = np.argmax(probabilities)
    confidence = np.max(probabilities)

    return predicted_digit, confidence


if __name__ == "__main__":
    image_path = "data/raw/sample_digit.png"

    digit, confidence = predict_digit(image_path)

    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.4f}")