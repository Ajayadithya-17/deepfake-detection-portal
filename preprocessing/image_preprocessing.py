"""
=============================================================================
Deepfake Detection System - Image Preprocessing & Augmentation Pipeline
=============================================================================
BSc Data Science Final-Year Project:
"AI-Based Deepfake Detection System Using Deep Learning"

Standardizes input images to the EfficientNet-B3 native input resolution
(300x300x3), applies ImageNet-calibrated normalization, and provides data
augmentation utilities for deep learning model training and evaluation.
=============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

TARGET_SIZE = (300, 300)

def preprocess_face_for_model(face_bgr, target_size=TARGET_SIZE):
    """
    Preprocesses a face image for the EfficientNet-B3 model.
    Returns an image with shape: (1, 300, 300, 3).
    """

    # Handle empty or invalid images
    if face_bgr is None or face_bgr.size == 0:
        return np.zeros(
            (1, target_size[0], target_size[1], 3),
            dtype=np.float32
        )

    # OpenCV loads images in BGR format, so convert to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Resize to EfficientNet-B3 input size
    face_rgb = cv2.resize(face_rgb, target_size)

    # Convert to float32
    face_rgb = face_rgb.astype(np.float32)

    # Apply EfficientNet preprocessing
    face_rgb = preprocess_input(face_rgb)

    # Add batch dimension
    face_rgb = np.expand_dims(face_rgb, axis=0)

    return face_rgb


def load_dataset_file_paths(dataset_dir):
    """
    Scans the dataset directory and returns file paths and binary labels:
    0 = REAL, 1 = FAKE.
    Supports:
        dataset/real/
        dataset/real/images/
        dataset/fake/
        dataset/fake/images/
    """

    file_paths = []
    labels = []

    valid_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return [], []

    # Search recursively for all image files
    for file_path in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, file_path)

        if not os.path.isdir(class_dir):
            continue

        class_name = file_path.lower().strip()

        # REAL = 0
        if class_name in {"real", "reals", "real_images", "authentic"}:
            label = 0

        # FAKE = 1
        elif class_name in {"fake", "fakes", "fake_images", "deepfake", "deepfakes"}:
            label = 1

        else:
            continue

        for image_path in os.walk(class_dir):
            current_root, _, filenames = image_path

            for filename in sorted(filenames):
                full_path = os.path.join(current_root, filename)

                if os.path.isfile(full_path):
                    extension = os.path.splitext(filename)[1].lower()

                    if extension in valid_exts:
                        file_paths.append(full_path)
                        labels.append(label)

    return file_paths, labels