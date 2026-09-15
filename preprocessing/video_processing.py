"""
=============================================================================
Deepfake Detection System - Video Processing & Frame Sequence Module
=============================================================================
BSc Data Science Final-Year Project:
"AI-Based Deepfake Detection System Using Deep Learning"

Extracts 20-30 equidistant keyframes across temporal video streams,
detects and crops facial regions, and formats sequential tensors
for CNN-LSTM temporal consistency analysis.
=============================================================================
"""

import os
import cv2
import numpy as np
from preprocessing.face_detection import detect_and_crop_face
from preprocessing.image_preprocessing import preprocess_face_for_model, TARGET_SIZE

DEFAULT_SEQUENCE_LENGTH = 20


def extract_video_face_sequence(video_path, num_frames=DEFAULT_SEQUENCE_LENGTH,
                                save_preview_dir=None, target_size=TARGET_SIZE):
    """
    Extracts an equidistant sequence of face crops from a video stream.

    Args:
        video_path (str): Path to input video file.
        num_frames (int): Number of frames to sample (default 20).
        save_preview_dir (str): Optional directory to save keyframe thumbnails.
        target_size (tuple): Resolution for model tensor (300, 300).

    Returns:
        tuple: (sequence_tensor, preview_paths, faces_found_count)
               - sequence_tensor: np.ndarray of shape (1, num_frames, 300, 300, 3)
               - preview_paths: List of saved preview image web paths.
               - faces_found_count: Number of frames where a clear face was localized.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video stream: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has invalid frame count ({total_frames})")

    # Determine equidistant frame indices across duration
    if total_frames <= num_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    processed_frames = []
    preview_paths = []
    faces_found = 0
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    for idx, frame_no in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ret, frame = cap.read()
        if not ret or frame is None:
            # If read error, duplicate previous or zero frame
            if processed_frames:
                processed_frames.append(processed_frames[-1].copy())
            continue

        # Extract face crop
        face_crop, bbox, found = detect_and_crop_face(frame, margin=0.20, target_size=target_size)
        if found:
            faces_found += 1

        # Preprocess for model: (1, 300, 300, 3) -> slice to (300, 300, 3)
        tensor_frame = preprocess_face_for_model(face_crop, target_size=target_size)[0]
        processed_frames.append(tensor_frame)

        # Save keyframe previews for UI timeline (up to 6 thumbnails)
        if save_preview_dir and len(preview_paths) < 6:
            os.makedirs(save_preview_dir, exist_ok=True)
            preview_filename = f"thumb_{video_basename}_f{idx}.jpg"
            disk_path = os.path.join(save_preview_dir, preview_filename)
            cv2.imwrite(disk_path, face_crop)
            preview_paths.append(f"/static/frames/{preview_filename}")

    cap.release()

    # Pad if video has fewer frames than expected
    while len(processed_frames) < num_frames:
        if processed_frames:
            processed_frames.append(processed_frames[-1].copy())
        else:
            processed_frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.float32))

    # Truncate to exact num_frames if exceeded
    processed_frames = processed_frames[:num_frames]

    # Stack into sequence tensor: (1, num_frames, 300, 300, 3)
    sequence_tensor = np.expand_dims(np.array(processed_frames, dtype=np.float32), axis=0)

    return sequence_tensor, preview_paths, faces_found

