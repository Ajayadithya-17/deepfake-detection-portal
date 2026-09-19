"""Train an EfficientNet-B3 classifier on Celeb-DF V2 face frames."""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input


TRAIN_DIR = "data/video_dataset/video"
MODEL_PATH = "models/video_frame_model.keras"
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 8
EPOCHS = 5
MAX_TRAIN_PER_CLASS = 5000
MAX_VAL_PER_CLASS = 1000
SEED = 42
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def collect_images(folder):
	if not os.path.exists(folder):
		return []
	return [
		os.path.join(root, filename)
		for root, _, filenames in os.walk(folder)
		for filename in filenames
		if filename.lower().endswith(SUPPORTED_EXTENSIONS)
	]


def prepare_dataset():
	def files(label, split):
		return collect_images(os.path.join(TRAIN_DIR, label, split))

	real_train, fake_train = files("real", "train"), files("fake", "train")
	real_val, fake_val = files("real", "val"), files("fake", "val")
	for values in (real_train, fake_train, real_val, fake_val):
		random.shuffle(values)

	real_train, fake_train = real_train[:MAX_TRAIN_PER_CLASS], fake_train[:MAX_TRAIN_PER_CLASS]
	real_val, fake_val = real_val[:MAX_VAL_PER_CLASS], fake_val[:MAX_VAL_PER_CLASS]
	train = [(p, 0) for p in real_train] + [(p, 1) for p in fake_train]
	val = [(p, 0) for p in real_val] + [(p, 1) for p in fake_val]
	random.shuffle(train)
	random.shuffle(val)
	if not train or not val:
		raise ValueError("Training and validation image folders must contain images.")
	return ([p for p, _ in train], [y for _, y in train],
			[p for p, _ in val], [y for _, y in val])


def load_image(path, label):
	image = tf.io.read_file(path)
	image = tf.image.decode_image(image, channels=3, expand_animations=False)
	image.set_shape([None, None, 3])
	image = tf.image.resize(image, IMAGE_SIZE)
	return preprocess_input(tf.cast(image, tf.float32)), label


def build_model():
	base = EfficientNetB3(weights="imagenet", include_top=False,
						  input_shape=(*IMAGE_SIZE, 3))
	freeze_until = int(len(base.layers) * 0.70)
	for layer in base.layers:
		layer.trainable = False
	for layer in base.layers[freeze_until:]:
		layer.trainable = True

	inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
	x = base(inputs, training=False)
	x = layers.GlobalAveragePooling2D()(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(256, activation="relu")(x)
	x = layers.Dropout(0.4)(x)
	outputs = layers.Dense(1, activation="sigmoid")(x)
	model = models.Model(inputs, outputs)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
		loss="binary_crossentropy",
		metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
				 tf.keras.metrics.Recall(name="recall")],
	)
	return model


def train():
	train_paths, train_labels, val_paths, val_labels = prepare_dataset()
	train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
	val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
	train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
	val_dataset = val_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
	augmentation = tf.keras.Sequential([
		layers.RandomFlip("horizontal"), layers.RandomRotation(0.05),
		layers.RandomZoom(0.10),
	])
	train_dataset = train_dataset.map(
		lambda x, y: (augmentation(x, training=True), y),
		num_parallel_calls=tf.data.AUTOTUNE,
	).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
	model = build_model()
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
										   save_best_only=True, verbose=1),
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
										 restore_best_weights=True, verbose=1),
		tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
											 patience=1, verbose=1),
	]
	model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS,
			  callbacks=callbacks)
	model.save(MODEL_PATH)
	print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
	train()
