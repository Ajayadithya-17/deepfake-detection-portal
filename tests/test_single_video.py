import sys
import os

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import cv2
import numpy as np
import tensorflow as tf

from preprocessing.face_detection import detect_and_crop_face
from preprocessing.image_preprocessing import (
    preprocess_face_for_model,
    TARGET_SIZE,
)

VIDEO_PATH = r"G:\deepfake_detection_portal\static\uploads\MAKE_A_VIDEO_BASED_ON_THIS_PIC.mp4"

MODEL_PATH = "models/video_frame_model.keras"

NUM_FRAMES = 20


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total video frames:", total_frames)

if total_frames <= 0:
    raise RuntimeError("Video contains no readable frames.")

# Select 20 equally spaced frames
indices = np.linspace(
    0,
    total_frames - 1,
    NUM_FRAMES,
    dtype=int
)

frames = []
valid_faces = 0

for frame_no in indices:

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_no)
    )

    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    face_crop, bbox, found = detect_and_crop_face(
        frame,
        margin=0.20,
        target_size=TARGET_SIZE,
    )

    if not found:
        continue

    tensor = preprocess_face_for_model(
        face_crop,
        target_size=TARGET_SIZE,
    )[0]

    frames.append(tensor)

    valid_faces += 1

cap.release()

if not frames:
    raise RuntimeError("No faces detected in the video.")

# If fewer than 20 faces were detected,
# duplicate the last valid frame to maintain 20 inputs.
while len(frames) < NUM_FRAMES:
    frames.append(frames[-1].copy())

frames = np.array(
    frames[:NUM_FRAMES],
    dtype=np.float32
)

print("Valid detected faces:", valid_faces)
print("Model input shape:", frames.shape)

# ============================================================
# MODEL PREDICTION
# ============================================================

predictions = model.predict(
    frames,
    verbose=0
).reshape(-1)

print("\n========== FRAME RESULTS ==========")

for i, p in enumerate(predictions, 1):

    print(
        f"Frame {i:02d}: "
        f"FAKE={p * 100:.2f}% "
        f"REAL={(1 - p) * 100:.2f}%"
    )

# ============================================================
# AGGREGATION COMPARISON
# ============================================================

mean_fake = float(np.mean(predictions))

median_fake = float(
    np.median(predictions)
)

sorted_predictions = np.sort(predictions)

top_5_count = min(5, len(predictions))
top_10_count = min(10, len(predictions))

top_5_mean = float(
    np.mean(sorted_predictions[-top_5_count:])
)

top_10_mean = float(
    np.mean(sorted_predictions[-top_10_count:])
)

max_fake = float(
    np.max(predictions)
)

frames_above_50 = int(
    np.sum(predictions >= 0.50)
)

frames_above_30 = int(
    np.sum(predictions >= 0.30)
)

# ============================================================
# FINAL DIAGNOSTIC OUTPUT
# ============================================================

print("\n========== AGGREGATION COMPARISON ==========")

print(
    f"Mean FAKE probability       : "
    f"{mean_fake * 100:.2f}%"
)

print(
    f"Median FAKE probability     : "
    f"{median_fake * 100:.2f}%"
)

print(
    f"Top-5 Mean FAKE probability : "
    f"{top_5_mean * 100:.2f}%"
)

print(
    f"Top-10 Mean FAKE probability: "
    f"{top_10_mean * 100:.2f}%"
)

print(
    f"Maximum FAKE probability    : "
    f"{max_fake * 100:.2f}%"
)

print(
    f"\nFrames >= 50% FAKE          : "
    f"{frames_above_50}/{len(predictions)}"
)

print(
    f"Frames >= 30% FAKE          : "
    f"{frames_above_30}/{len(predictions)}"
)

print("============================================")

# ============================================================
# CURRENT METHOD
# ============================================================

current_prediction = (
    "DEEPFAKE"
    if mean_fake >= 0.50
    else "REAL"
)

print(
    f"\nCURRENT MEAN PREDICTION: "
    f"{current_prediction}"
)

# ============================================================
# DIAGNOSTIC ONLY
# ============================================================
# This is NOT being used in Flask yet.
# We are only checking what the Top-5 aggregation would produce.

top_5_prediction = (
    "DEEPFAKE"
    if top_5_mean >= 0.50
    else "REAL"
)

print(
    f"TOP-5 DIAGNOSTIC PREDICTION: "
    f"{top_5_prediction}"
)

print("\n========== END TEST ==========")