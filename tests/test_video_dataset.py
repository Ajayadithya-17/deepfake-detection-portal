"""
Video-Level Evaluation using Celeb-DF V2 Cropped Frames

Each folder represents one source video.
We sample up to 20 frames from each folder, run the trained
EfficientNet-B3 image model, average the frame probabilities,
and produce one prediction for the entire video.
"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from preprocessing.image_preprocessing import preprocess_face_for_model


MODEL_PATH = "models/video_frame_model.keras"
FAKE_TEST_DIR = "data/video_dataset/video/fake/test"
REAL_TEST_DIR = "data/video_dataset/video/real/test"
FRAMES_PER_VIDEO = 20
REAL_LABEL = 0
FAKE_LABEL = 1


print("\nLoading trained EfficientNet-B3 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def get_frame_files(folder):
    files = [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    files.sort()
    return files


def predict_video_folder(folder_path):
    frame_files = get_frame_files(folder_path)
    if not frame_files:
        return None

    if len(frame_files) > FRAMES_PER_VIDEO:
        indices = np.linspace(0, len(frame_files) - 1, FRAMES_PER_VIDEO, dtype=int)
        selected_files = [frame_files[i] for i in indices]
    else:
        selected_files = frame_files

    processed_frames = []
    for frame_path in selected_files:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        try:
            processed_frames.append(preprocess_face_for_model(frame)[0])
        except Exception as error:
            print(f"Warning: Could not process {frame_path}: {error}")

    if not processed_frames:
        return None

    probabilities = model.predict(
        np.array(processed_frames, dtype=np.float32), verbose=0
    ).flatten()
    average_fake_probability = float(np.mean(probabilities))
    predicted_label = (
        FAKE_LABEL if average_fake_probability >= 0.5 else REAL_LABEL
    )

    return {
        "prediction": predicted_label,
        "fake_probability": average_fake_probability,
        "frames_used": len(processed_frames),
    }


def evaluate_video_dataset():
    y_true = []
    y_pred = []
    fake_correct = fake_total = real_correct = real_total = 0

    print("\n==============================================")
    print("VIDEO-LEVEL DEEPFAKE EVALUATION")
    print("==============================================")

    for label, directory, name in (
        (FAKE_LABEL, FAKE_TEST_DIR, "Fake"),
        (REAL_LABEL, REAL_TEST_DIR, "Real"),
    ):
        folders = [
            os.path.join(directory, item)
            for item in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, item))
        ]
        print(f"\n{name} videos found: {len(folders)}")

        for index, folder in enumerate(folders, 1):
            result = predict_video_folder(folder)
            if result is None:
                continue
            predicted = result["prediction"]
            y_true.append(label)
            y_pred.append(predicted)
            if label == FAKE_LABEL:
                fake_total += 1
                fake_correct += predicted == label
            else:
                real_total += 1
                real_correct += predicted == label
            if index % 25 == 0:
                print(f"Processed {name.lower()} videos: {index}/{len(folders)}")

    print("\n==============================================")
    print("FINAL VIDEO-LEVEL RESULTS")
    print("==============================================")
    metrics = (
        ("Video Accuracy", accuracy_score(y_true, y_pred)),
        ("Video Precision", precision_score(y_true, y_pred, zero_division=0)),
        ("Video Recall", recall_score(y_true, y_pred, zero_division=0)),
        ("Video F1-Score", f1_score(y_true, y_pred, zero_division=0)),
    )
    print(f"\nTotal videos evaluated : {len(y_true)}")
    print(f"Fake videos evaluated  : {fake_total}")
    print(f"Real videos evaluated  : {real_total}")
    for metric_name, value in metrics:
        print(f"{metric_name:<24}: {value * 100:.2f}%")

    print("\n==============================================")
    print("CLASSIFICATION REPORT")
    print("==============================================")
    print(classification_report(
        y_true, y_pred, target_names=["REAL", "DEEPFAKE"], zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=[REAL_LABEL, FAKE_LABEL])
    print("==============================================")
    print("CONFUSION MATRIX")
    print("==============================================")
    print("\n              Predicted")
    print("              REAL   FAKE")
    print(f"Actual REAL   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"Actual FAKE   {cm[1][0]:5d}  {cm[1][1]:5d}")
    print("\n==============================================")
    print("EVALUATION COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    evaluate_video_dataset()