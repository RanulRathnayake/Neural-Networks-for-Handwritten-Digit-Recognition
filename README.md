# Handwritten Digit Recognition using Neural Networks and MLflow

This project is an end-to-end **Handwritten Digit Recognition** system built using **TensorFlow/Keras**, **MLflow**, and **FastAPI**.

The model is trained on the **MNIST handwritten digit dataset** to classify digits from `0` to `9`. The project includes model training, experiment tracking with MLflow, model saving/loading, and a backend API for real-time image prediction.

---

## Project Overview

The main goal of this project is to convert a neural network assignment into a complete machine learning project with a proper workflow.

This project includes:

- Handwritten digit classification using a neural network
- MNIST dataset loading and preprocessing
- Model training using TensorFlow/Keras
- Experiment tracking using MLflow
- Model saving locally using `.keras` format
- Model logging inside MLflow artifacts
- FastAPI backend for image upload and prediction
- Swagger UI testing for the prediction API

---

## Tech Stack

- Python
- TensorFlow / Keras
- MLflow
- FastAPI
- NumPy
- Pillow
- Uvicorn
- MNIST Dataset

---

## Project Structure

```text
handwritten-digit-recognition/
│
├── app/
│   └── api.py
│
├── artifacts/
│   ├── model_summary.txt
│   └── model_architecture.json
│
├── data/
│   └── raw/
│       └── sample_digit.png
│
├── mlruns/
│
├── models/
│   └── digit_model.keras
│
├── notebooks/
│   └── Handwritten Digit Recognition Assignment.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

### 1. Data Loading and Preprocessing

The project uses the MNIST dataset from TensorFlow.

Preprocessing steps:

- Load MNIST train and test data
- Normalize pixel values from `0-255` to `0-1`
- Flatten each image from `28 x 28` into `784` input features

---

### 2. Neural Network Model

The model is a dense neural network for multiclass classification.

Example architecture:

```text
Input Layer: 784 features
Hidden Layer 1: Dense layer with ReLU activation
Hidden Layer 2: Dense layer with ReLU activation
Output Layer: 10 neurons for digits 0-9
```

The output layer uses `linear` activation because the loss function uses:

```python
SparseCategoricalCrossentropy(from_logits=True)
```

Softmax is applied during prediction to convert logits into probabilities.

---

### 3. Model Training

The model is trained using:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy
- Dataset: MNIST
- Batch Size: 32
- Maximum Epochs: 50
- EarlyStopping enabled to avoid overfitting

EarlyStopping monitors validation loss and stops training if the model does not improve.

---

### 4. MLflow Experiment Tracking

MLflow is used to track the training process.

The project logs:

- Parameters
  - Maximum epochs
  - Actual trained epochs
  - Batch size
  - Learning rate
  - Optimizer
  - Loss function
  - Dataset
  - Model type
  - Total model parameters

- Metrics
  - Training loss
  - Training accuracy
  - Validation loss
  - Validation accuracy
  - Test loss
  - Test accuracy

- Artifacts
  - Trained TensorFlow/Keras model
  - Model summary
  - Model architecture JSON

---

## Setup Instructions


### 1. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.


```bash
venv\Scripts\activate
```
---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Train the Model

Run the training script from the project root:

```bash
python src/train.py
```

After training, the model will be saved locally:

```text
models/digit_model.keras
```

The trained model will also be logged inside MLflow artifacts.

---

## How to Open MLflow UI

Start MLflow UI from the project root:

```bash
python -m mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

Go to:

```text
Model training → Experiments → Handwritten Digit Recognition → Runs
```

Inside the run, you can check:

- Overview
- Model metrics
- Parameters
- Artifacts

---

## How to Test Prediction Locally

Run:

```bash
python src/predict.py
```

This script loads the saved model and predicts a digit from an image.

Make sure the test image exists in:

```
data/raw/sample_digit.png
```

---

## FastAPI Backend

The project includes a FastAPI backend for image prediction.

The API accepts an uploaded digit image and returns:

- Predicted digit
- Confidence score

---

## How to Run the API

Run this command from the project root:

```bash
uvicorn app.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

You should see:

```json
{
  "message": "Handwritten Digit Recognition API is running"
}
```

---

## API Documentation

FastAPI automatically provides Swagger UI.

Open:

```text
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint:

1. Click `/predict`
2. Click `Try it out`
3. Choose an image file
4. Click `Execute`
5. Check the predicted digit and confidence

---

## API Endpoint

### POST `/predict`

Request type:

```text
multipart/form-data
```

Form field:

```text
file
```

Example response:

```json
{
  "predicted_digit": 3,
  "confidence": 0.9509409070014954
}
```


---

## Model Performance

Example results from MLflow:

```text
Test Accuracy: 0.9793
Test Loss: 0.0719
Training Accuracy: 0.9842
Validation Accuracy: 0.9793
```

The results may change slightly depending on training runs, random initialization, and the number of epochs completed before EarlyStopping.

---

## Screenshots

### MLflow Run Overview



### MLflow Model Metrics



### FastAPI Swagger Upload



### FastAPI Prediction Response

---

## Author

**Ranul Rathnayake**

Machine Learning / Deep Learning Project  
Handwritten Digit Recognition using TensorFlow, MLflow, and FastAPI
