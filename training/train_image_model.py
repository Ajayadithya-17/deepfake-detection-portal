"""
=============================================================================
Deepfake Detection System - EfficientNet-B3 Image Model Training
=============================================================================
BSc Data Science Final-Year Project:
"AI-Based Deepfake Detection System Using Deep Learning"

Trains an EfficientNet-B3 convolutional neural network with transfer learning
for binary deepfake classification (REAL=0, DEEPFAKE=1).
Implements 70% Training / 15% Validation / 15% Testing partitioning,
data augmentation, EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
Outputs Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
=============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.applications.efficientnet import preprocess_input

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3
from preprocessing.image_preprocessing import preprocess_face_for_model, TARGET_SIZE, load_dataset_file_paths

TARGET_SIZE = (300, 300)


MODEL_SAVE_PATH = os.path.join('models', 'image_model.keras')
CM_SAVE_PATH = os.path.join('models', 'image_confusion_matrix.png')


def build_efficientnet_b3_classifier(input_shape=(300, 300, 3), learning_rate=1e-4):
    """
    Constructs the transfer learning architecture using EfficientNet-B3 backbone.
    """
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Freeze the first 70% of base model layers for feature extraction stability
    for layer in base_model.layers[:int(len(base_model.layers) * 0.70)]:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_face")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation='relu', name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout_regularizer")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="deepfake_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB3_Deepfake_Detector")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model


def train_image_model(dataset_dir='data/image_dataset', epochs=5, batch_size=8):

    """
    Loads dataset, partitions into 70/15/15 splits, trains model, and generates evaluation metrics.
    """
    os.makedirs('models', exist_ok=True)
    print("==================================================================")
    print("Step 1: Loading Dataset & Splitting (70% Train, 15% Val, 15% Test)")
    print("==================================================================")

    file_paths, labels = load_dataset_file_paths("data/image_dataset")

    real_paths = [p for p, y in zip(file_paths, labels) if y == 0][:5000]
    fake_paths = [p for p, y in zip(file_paths, labels) if y == 1][:5000]

    file_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)

    print(f"Training dataset selected: {len(file_paths)}")
    print(f"REAL: {labels.count(0)}")
    print(f"FAKE: {labels.count(1)}")

    file_paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)

    print(f"Quick test dataset: {len(file_paths)} images")
    print(f"REAL: {labels.count(0)}")
    print(f"FAKE: {labels.count(1)}")

    if len(file_paths) == 0:
        raise ValueError(f"No valid images found in {dataset_dir}. Please ensure dataset/real/images and dataset/fake/images exist.")

    print(f"Total dataset samples localized: {len(file_paths)} (Real: {labels.count(0)}, Fake: {labels.count(1)})")

    # 70% Train, 30% Temp (which will be split 50/50 into Val and Test -> 15% each)
    train_paths, temp_paths, train_y, temp_y = train_test_split(
        file_paths, labels, test_size=0.30, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        temp_paths, temp_y, test_size=0.50, random_state=42, stratify=temp_y
    )

    print(f"Split distribution: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    def data_generator(paths, y_labels, batch_sz=batch_size, is_training=True):
        num_samples = len(paths)
        while True:
            indices = np.arange(num_samples)
            if is_training:
                np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_sz):
                batch_indices = indices[start_idx:start_idx + batch_sz]
                batch_x = []
                batch_y = []
                for i in batch_indices:
                    img = cv2.imread(paths[i])
                    if img is None:
                        continue
                    processed = preprocess_face_for_model(img, target_size=TARGET_SIZE)[0]
                    batch_x.append(processed)
                    batch_y.append(y_labels[i])
                if batch_x:
                    yield np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    train_gen = data_generator(train_paths, train_y, batch_sz=batch_size, is_training=True)
    val_gen = data_generator(val_paths, val_y, batch_sz=batch_size, is_training=False)

    train_steps = math.ceil(len(train_paths) / batch_size)
    val_steps = math.ceil(len(val_paths) / batch_size)

    print("\nStep 2: Compiling EfficientNet-B3 Model Architecture")
    model = build_efficientnet_b3_classifier()
    model.summary()

    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss",patience=4,min_delta=0.001,restore_best_weights=True,verbose=1),
        callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    print("\nStep 3: Training Deep Learning Model with Callbacks")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=training_callbacks
    )

    print("\nStep 4: Evaluating on Independent Test Set (15%)")
    test_x = []
    test_true = []
    for p, y in zip(test_paths, test_y):
        img = cv2.imread(p)
        if img is not None:
            test_x.append(preprocess_face_for_model(img, target_size=TARGET_SIZE)[0])
            test_true.append(y)

    test_x = np.array(test_x, dtype=np.float32)
    test_true = np.array(test_true, dtype=np.int32)

    predictions = model.predict(test_x)
    pred_classes = (predictions >= 0.5).astype(int).flatten()

    acc = accuracy_score(test_true, pred_classes)
    prec = precision_score(test_true, pred_classes, zero_division=0)
    rec = recall_score(test_true, pred_classes, zero_division=0)
    f1 = f1_score(test_true, pred_classes, zero_division=0)

    print("\n==================== TEST SET PERFORMANCE METRICS ====================")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(test_true, pred_classes, target_names=['REAL', 'DEEPFAKE'], zero_division=0))

    # Confusion Matrix Visualization
    cm = confusion_matrix(test_true, pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'DEEPFAKE'], yticklabels=['REAL', 'DEEPFAKE'])
    plt.title('EfficientNet-B3 Deepfake Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth Label')
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH)
    plt.close()
    print(f"Confusion matrix plot saved to {CM_SAVE_PATH}")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_image_model(epochs=5, batch_size=8)

