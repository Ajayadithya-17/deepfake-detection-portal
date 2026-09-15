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
    Prepares a cropped BGR face image for EfficientNet-B3 model inference.

    Args:
        face_bgr (np.ndarray): Cropped face image in BGR format.
        target_size (tuple): Target (height, width), defaults to (300, 300).

    Returns:
        np.ndarray: Preprocessed batch tensor of shape (1, 300, 300, 3).
    """
    if face_bgr is None or face_bgr.size == 0:
        return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Resize to EfficientNet-B3 input resolution
    if (face_rgb.shape[0], face_rgb.shape[1]) != target_size:
        face_rgb = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Convert to float array
    img_array = np.array(face_rgb, dtype=np.float32)

    # Apply EfficientNet-B3 preprocessing
    img_array = preprocess_input(img_array)

    # Expand batch dimension: (1, 300, 300, 3)
    return np.expand_dims(img_array, axis=0)


def get_data_augmentation_layers():
    """
    Builds a Keras sequential data augmentation pipeline for training.
    Applies spatial perturbations to prevent overfitting on specific lighting
    and compression artifacts.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ], name="data_augmentation")


def load_dataset_file_paths(dataset_dir):
    """
    Scans the dataset directory and returns file paths and binary labels:
    0 = REAL, 1 = FAKE (DEEPFAKE).

    Args:
        dataset_dir (str): Root dataset directory containing 'real/images' and 'fake/images'.

    Returns:
        tuple: (file_paths_list, labels_list)
    """
    file_paths = []
    labels = []

    real_dir = os.path.join(dataset_dir, 'real', 'images')
    fake_dir = os.path.join(dataset_dir, 'fake', 'images')

    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')

    if os.path.exists(real_dir):
        for f in sorted(os.listdir(real_dir)):
            if f.lower().endswith(valid_exts):
                file_paths.append(os.path.join(real_dir, f))
                labels.append(0)  # Real

    if os.path.exists(fake_dir):
        for f in sorted(os.listdir(fake_dir)):
            if f.lower().endswith(valid_exts):
                file_paths.append(os.path.join(fake_dir, f))
                labels.append(1)  # Deepfake

    return file_paths, labels

