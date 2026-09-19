"""
=============================================================================
Deepfake Detection System - CNN-LSTM Temporal Video Model Training
=============================================================================
BSc Data Science Final-Year Project:
"AI-Based Deepfake Detection System Using Deep Learning"

Builds and trains a hybrid CNN-LSTM architecture:
1. Feature Extractor: Pretrained EfficientNet-B3 (1536-dim spatial representation)
2. Sequence Analyzer: Bidirectional LSTM capturing inter-frame facial anomalies,
   flickering, and temporal inconsistency across 20-frame sequences.
Outputs Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
=============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3
from preprocessing.video_processing import extract_video_face_sequence, DEFAULT_SEQUENCE_LENGTH

VIDEO_MODEL_SAVE_PATH = os.path.join('models', 'video_model.keras')
VIDEO_CM_SAVE_PATH = os.path.join('models', 'video_confusion_matrix.png')


def build_cnn_lstm_video_model(sequence_length=DEFAULT_SEQUENCE_LENGTH, feature_dim=1536, learning_rate=1e-4):
    """
    Constructs the CNN-LSTM temporal sequence classification architecture.
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name="frame_feature_sequence")

    # Temporal sequence modeling via Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2),
        name="bidirectional_lstm"
    )(inputs)
    x = layers.BatchNormalization(name="seq_batch_norm")(x)
    x = layers.Dense(64, activation='relu', name="dense_temporal")(x)
    x = layers.Dropout(0.3, name="dropout_temporal")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="video_deepfake_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Video_Deepfake_Detector")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model


def extract_features_from_sequence(extractor, sequence_tensor):
    """
    Passes (1, num_frames, 300, 300, 3) through EfficientNet-B3 to obtain
    a feature vector sequence of shape (1, num_frames, 1536).
    """
    frames = sequence_tensor[0]
    features = extractor.predict(frames, verbose=0)
    return np.expand_dims(features, axis=0)  # (1, num_frames, 1536)


def train_video_model(dataset_dir='data/video_dataset', epochs=5):
    """
    Gathers video dataset, extracts temporal sequences, extracts features,
    and trains the Bidirectional LSTM model.
    """
    os.makedirs('models', exist_ok=True)
    print("==================================================================")
    print("Step 1: Discovering Video Dataset Samples")
    print("==================================================================")

    real_videos_dir = os.path.join(dataset_dir, 'real', 'videos')
    fake_videos_dir = os.path.join(dataset_dir, 'fake', 'videos')

    video_paths = []
    labels = []

    valid_exts = ('.mp4', '.avi', '.mov')
    if os.path.exists(real_videos_dir):
        for f in sorted(os.listdir(real_videos_dir)):
            if f.lower().endswith(valid_exts):
                video_paths.append(os.path.join(real_videos_dir, f))
                labels.append(0)  # Real

    if os.path.exists(fake_videos_dir):
        for f in sorted(os.listdir(fake_videos_dir)):
            if f.lower().endswith(valid_exts):
                video_paths.append(os.path.join(fake_videos_dir, f))
                labels.append(1)  # Deepfake

    print(f"Total video samples located: {len(video_paths)} (Real: {labels.count(0)}, Fake: {labels.count(1)})")

    if len(video_paths) == 0:
        print("No video files found in dataset. Building and saving initialized model weights.")
        model = build_cnn_lstm_video_model()
        model.save(VIDEO_MODEL_SAVE_PATH)
        print(f"Saved initialized model to {VIDEO_MODEL_SAVE_PATH}")
        return

    print("\nStep 2: Initializing Pretrained EfficientNet-B3 Feature Extractor")
    base_cnn = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg', input_shape=(300, 300, 3))

    features_list = []
    y_list = []

    for vpath, label in zip(video_paths, labels):
        try:
            print(f"  Extracting 20-frame sequence from: {os.path.basename(vpath)}")
            seq_tensor, _, _ = extract_video_face_sequence(vpath, num_frames=DEFAULT_SEQUENCE_LENGTH)
            feat_seq = extract_features_from_sequence(base_cnn, seq_tensor)  # (1, 20, 1536)
            features_list.append(feat_seq[0])
            y_list.append(label)
        except Exception as e:
            print(f"Skipping {vpath} due to read error: {e}")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Feature dataset shape: X={X.shape}, y={y.shape}")

    # Build sequence model
    model = build_cnn_lstm_video_model()
    model.summary()

    if len(X) >= 4:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=2)

        preds = (model.predict(X_test) >= 0.5).astype(int).flatten()
        print("\nEvaluation Metrics:")
        print(classification_report(y_test, preds, zero_division=0))
    else:
        model.fit(X, y, epochs=epochs, batch_size=2)

    model.save(VIDEO_MODEL_SAVE_PATH)
    print(f"Saved video model to {VIDEO_MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_video_model(epochs=5)

