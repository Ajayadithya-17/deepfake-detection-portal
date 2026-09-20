"""
=============================================================================
Deepfake Detection System - Video Processing & Frame Extraction Module
=============================================================================
BSc Data Science Final-Year Project:
"AI-Based Deepfake Detection System Using Deep Learning"

Extracts 20 equidistant frames from a video, detects and crops facial
regions, and prepares frames for EfficientNet-B3 frame-level classification.
=============================================================================
"""

import os
import cv2
import numpy as np

from preprocessing.face_detection import detect_and_crop_face
from preprocessing.image_preprocessing import (
    preprocess_face_for_model,
    TARGET_SIZE
)

DEFAULT_SEQUENCE_LENGTH = 20


def extract_video_face_sequence(
    video_path,
    num_frames=DEFAULT_SEQUENCE_LENGTH,
    save_preview_dir=None,
    target_size=TARGET_SIZE
):
    """
    Extract 20 equidistant video frames and prepare them for
    EfficientNet-B3 frame-level classification.

    Returns:
        sequence_tensor:
            Shape (1, num_frames, 300, 300, 3)

        preview_paths:
            Saved preview image paths.

        faces_found_count:
            Number of sampled frames where a face was detected.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video stream: {video_path}"
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(
            f"Video has invalid frame count ({total_frames})"
        )

    # ---------------------------------------------------------
    # Select equidistant frames
    # ---------------------------------------------------------

    if total_frames <= num_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(
            0,
            total_frames - 1,
            num_frames,
            dtype=int
        )

    processed_frames = []
    preview_paths = []
    faces_found = 0

    video_basename = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    # ---------------------------------------------------------
    # Process sampled frames
    # ---------------------------------------------------------

    for idx, frame_no in enumerate(indices):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_no)
        )

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # Detect and crop face
        face_crop, bbox, found = detect_and_crop_face(
            frame,
            margin=0.40,
            target_size=target_size
        )

        if not found:
            # Do NOT count this as a valid face frame.
            continue

        faces_found += 1

        # Preprocess detected face
        tensor_frame = preprocess_face_for_model(
            face_crop,
            target_size=target_size
        )[0]

        processed_frames.append(tensor_frame)

        # -----------------------------------------------------
        # Save preview
        # -----------------------------------------------------

        if (
            save_preview_dir
            and len(preview_paths) < 6
        ):
            os.makedirs(
                save_preview_dir,
                exist_ok=True
            )

            preview_filename = (
                f"thumb_{video_basename}_f{idx}.jpg"
            )

            disk_path = os.path.join(
                save_preview_dir,
                preview_filename
            )

            cv2.imwrite(
                disk_path,
                face_crop
            )

            preview_paths.append(
                f"/static/frames/{preview_filename}"
            )

    cap.release()

    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------

    if not processed_frames:
        raise ValueError(
            "No valid face frames were detected in the video."
        )

    # ---------------------------------------------------------
    # Use only valid face frames
    # ---------------------------------------------------------
    # Do NOT duplicate frames when fewer than 20 faces are found.
    # The video model performs frame-level classification, so
    # averaging only genuine detected-face frames avoids giving
    # artificial weight to a single frame.
    
    if not processed_frames:
        raise ValueError(
            "No valid face frames were detected in the video."
    )

# Use all valid detected-face frames.
# No duplicate padding is performed.

    # ---------------------------------------------------------
    # Create model tensor
    # ---------------------------------------------------------

    sequence_tensor = np.expand_dims(
        np.array(
            processed_frames,
            dtype=np.float32
        ),
        axis=0
    )

    return (
        sequence_tensor,
        preview_paths,
        faces_found
    )