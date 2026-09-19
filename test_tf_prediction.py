import os
import random
from utils.tf_predictor import predict_image

REAL_DIR = r"data/image_dataset/real"
FAKE_DIR = r"data/image_dataset/fake"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def get_random_images(folder, limit=100):
    files = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.lower().endswith(VALID_EXTENSIONS)
    ]

    random.shuffle(files)
    return files[:limit]


def test_folder(folder, expected_label, limit=100):
    files = get_random_images(folder, limit)

    correct = 0
    false_predictions = []

    print(f"\nTesting: {folder}")

    for file_path in files:
        result = predict_image(file_path)

        predicted_label = result["verdict"]
        fake_probability = result["fake_probability"]

        if predicted_label == expected_label:
            correct += 1
        else:
            false_predictions.append(
                (os.path.basename(file_path), fake_probability)
            )

    accuracy = correct / len(files) if files else 0

    print(f"Correct: {correct}/{len(files)}")
    print(f"Accuracy: {accuracy:.2%}")

    if false_predictions:
        print("\nIncorrect predictions:")
        for filename, probability in false_predictions[:15]:
            print(f"{filename} → Fake probability: {probability:.2%}")

    return correct, len(files)


real_correct, real_total = test_folder(
    REAL_DIR,
    "REAL",
    limit=100
)

fake_correct, fake_total = test_folder(
    FAKE_DIR,
    "DEEPFAKE",
    limit=100
)

total_correct = real_correct + fake_correct
total_images = real_total + fake_total

print("\n==============================")
print("FINAL RANDOM TEST RESULT")
print("==============================")
print(f"Correct: {total_correct}/{total_images}")
print(f"Overall accuracy: {total_correct / total_images:.2%}")