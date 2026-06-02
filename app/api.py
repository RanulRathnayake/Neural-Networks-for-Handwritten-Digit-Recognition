import sys
import os
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = FastAPI(title="Handwritten Digit Recognition API")

model = tf.keras.models.load_model("models/digit_model.keras")


def preprocess_uploaded_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = 1.0 - image_array
    image_array = image_array.reshape(1, 28 * 28)

    return image_array


@app.get("/")
def home():
    return {"message": "Handwritten Digit Recognition API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)

    processed_image = preprocess_uploaded_image(image)

    logits = model.predict(processed_image)
    probabilities = tf.nn.softmax(logits).numpy()[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence
    }