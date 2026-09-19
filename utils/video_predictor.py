import cv2
import numpy as np
import tensorflow as tf

from preprocessing.image_preprocessing import preprocess_face_for_model


MODEL_PATH = "models/image_model.keras"

# Load the same model used for image prediction.
model = tf.keras.models.load_model(MODEL_PATH)


def predict_video(video_path, num_frames=20):
	"""Predict a video using frame-wise inference from the image model."""
	if num_frames <= 0:
		raise ValueError("num_frames must be greater than zero.")

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError("Could not open the video file.")

	try:
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if total_frames <= 0:
			raise ValueError("Video contains no readable frames.")

		frame_indices = np.linspace(
			0,
			total_frames - 1,
			min(num_frames, total_frames),
			dtype=int,
		)

		probabilities = []
		for frame_index in frame_indices:
			cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
			success, frame = cap.read()
			if not success:
				continue

			try:
				processed_frame = preprocess_face_for_model(frame)
				prediction = model.predict(processed_frame, verbose=0)
				probabilities.append(float(prediction[0][0]))
			except Exception as exc:
				print(f"Frame {frame_index} skipped: {exc}")

		if not probabilities:
			raise ValueError("No frames could be analyzed.")

		average_fake_probability = float(np.mean(probabilities))
		return {
			"verdict": (
				"DEEPFAKE" if average_fake_probability >= 0.5 else "REAL"
			),
			"fake_probability": average_fake_probability * 100,
			"real_probability": (1 - average_fake_probability) * 100,
			"frames_analyzed": len(probabilities),
			"total_frames": total_frames,
		}
	finally:
		cap.release()
