import os
import json
import tempfile

import mlflow
import mlflow.tensorflow

from data_loader import load_mnist_data
from model import build_model


def get_model_summary(model):
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    return "\n".join(summary_lines)


def train():
    X_train, y_train, X_test, y_test = load_mnist_data()

    epochs = 50
    batch_size = 32
    learning_rate = 0.001

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Handwritten Digit Recognition")

    with mlflow.start_run(run_name="dense-neural-network-mnist"):

        model = build_model(
            input_shape=784,
            learning_rate=learning_rate
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test)

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_function": "SparseCategoricalCrossentropy",
            "model_type": "Dense Neural Network",
            "dataset": "MNIST",
            "input_shape": 784,
            "output_classes": 10,
            "total_parameters": model.count_params()
        })

        mlflow.log_metrics({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy)
        })

        for epoch in range(epochs):
            mlflow.log_metric("train_loss", float(history.history["loss"][epoch]), step=epoch)
            mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch]), step=epoch)
            mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)

        os.makedirs("models", exist_ok=True)
        model.save("models/digit_model.keras")

        mlflow.tensorflow.log_model(
            model,
            artifact_path="digit_recognition_model"
        )

        model_summary = get_model_summary(model)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(model_summary)
            summary_path = f.name

        mlflow.log_artifact(summary_path, artifact_path="model_details")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8"
        ) as f:
            json.dump(json.loads(model.to_json()), f, indent=4)
            architecture_path = f.name

        mlflow.log_artifact(architecture_path, artifact_path="model_details")

        print("Training completed.")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("MLflow run saved successfully.")
        print("Model saved locally and inside MLflow.")


if __name__ == "__main__":
    train()