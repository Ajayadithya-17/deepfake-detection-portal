"""
=============================================================================
Deepfake Detection System - Face Detection & Cropping Module
=============================================================================
BSc Data Science Final-Year Project:
"AI-Based Deepfake Detection System Using Deep Learning"

This module isolates the primary facial region using MTCNN (Multi-task
Cascaded Convolutional Networks) with adaptive boundary padding to capture
subtle blending and warping artifacts at facial perimeters. It includes a
fallback to OpenCV Haar Cascade and square center-cropping.
=============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN detector
try:
    detector = MTCNN()
except Exception:
    detector = None

# Fallback Haar Cascade (Safely check candidate locations without relying unconditionally on cv2.data)
face_cascade = None
candidate_paths = [
    os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'haarcascade_frontalface_default.xml'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessing', 'haarcascade_frontalface_default.xml'),
]
if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
    candidate_paths.append(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))

for cp in candidate_paths:
    if os.path.exists(cp):
        try:
            cascade = cv2.CascadeClassifier(cp)
            if not cascade.empty():
                face_cascade = cascade
                break
        except Exception:
            pass


def detect_and_crop_face(image_bgr, margin=0.20, target_size=(300, 300)):
    """
    Detects the primary face in an image and crops it with boundary margin padding.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        margin (float): Fraction of bounding box dimension added as padding (default 20%).
        target_size (tuple): Optional resize dimensions (width, height) for model input.

    Returns:
        tuple: (cropped_face, bbox, face_found)
               - cropped_face: Cropped BGR face image.
               - bbox: (x, y, w, h) bounding box coordinates.
               - face_found: True if a human face was detected, False if fallback was used.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, None, False

    img_h, img_w = image_bgr.shape[:2]

    # Pre-scale large images for sub-100ms detection latency
    max_dim = max(img_h, img_w)
    scale = 800.0 / max_dim if max_dim > 800 else 1.0
    if scale < 1.0:
        det_img = cv2.resize(image_bgr, (int(img_w * scale), int(img_h * scale)))
    else:
        det_img = image_bgr

    detected_box = None

    # Step 1: MTCNN Detection
    if detector is not None:
        try:
            det_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(det_rgb)
            if detections and len(detections) > 0:
                best_det = max(detections, key=lambda d: d.get('confidence', 0))
                if best_det.get('confidence', 0) > 0.70:
                    bx, by, bw, bh = best_det['box']
                    detected_box = (
                        int(max(0, bx) / scale),
                        int(max(0, by) / scale),
                        int(bw / scale),
                        int(bh / scale)
                    )
        except Exception:
            detected_box = None

    # Step 2: OpenCV Haar Cascade Fallback
    if detected_box is None and face_cascade is not None:
        try:
            gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                detected_box = (
                    int(fx / scale),
                    int(fy / scale),
                    int(fw / scale),
                    int(fh / scale)
                )
        except Exception:
            detected_box = None

    # Step 3: Crop with 20% boundary padding
    if detected_box is not None:
        x, y, w, h = detected_box
        pad_w = int(w * margin)
        pad_h = int(h * margin)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            if target_size is not None:
                crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            return crop, (x1, y1, x2 - x1, y2 - y1), True

    # Step 4: Aspect-Ratio Preserving Square Center-Crop Fallback
    crop_size = min(img_h, img_w)
    cy, cx = img_h // 2, img_w // 2
    y1 = max(0, cy - crop_size // 2)
    y2 = min(img_h, y1 + crop_size)
    x1 = max(0, cx - crop_size // 2)
    x2 = min(img_w, x1 + crop_size)
    crop = image_bgr[y1:y2, x1:x2]
    if target_size is not None:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return crop, (x1, y1, crop_size, crop_size), False

