import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("models", "image_model.keras")
TARGET_SIZE = (300, 300)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_bgr):
    """
    Use exactly the same preprocessing as the current training pipeline.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image")

    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize to EfficientNet-B3 input size
    image_rgb = cv2.resize(image_rgb, TARGET_SIZE)

    # Float32
    image_rgb = image_rgb.astype(np.float32)

    # EfficientNet preprocessing
    image_rgb = tf.keras.applications.efficientnet.preprocess_input(
        image_rgb
    )

    # Add batch dimension
    return np.expand_dims(image_rgb, axis=0)


def predict_image(image_path):
    """
    Predict REAL or DEEPFAKE using the same input pipeline
    used during the current model evaluation.
    """

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    input_tensor = preprocess_image(image_bgr)

    prediction = model.predict(
        input_tensor,
        verbose=0
    )

    fake_probability = float(prediction[0][0])
    real_probability = 1.0 - fake_probability

    verdict = (
        "DEEPFAKE"
        if fake_probability >= 0.5
        else "REAL"
    )

    return {
        "verdict": verdict,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }