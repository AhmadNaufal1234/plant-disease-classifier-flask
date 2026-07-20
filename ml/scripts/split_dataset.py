import os
import random
import shutil
from pathlib import Path

# ============================
# KONFIGURASI
# ============================
TRAIN_RATIO = 0.8  # 80% training, 20% testing
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ============================
# PATH
# ============================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
TRAIN_DIR = BASE_DIR / "dataset" / "training"
TEST_DIR = BASE_DIR / "dataset" / "testing"

# ============================
# HAPUS FOLDER LAMA
# ============================
for folder in [TRAIN_DIR, TEST_DIR]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

# ============================
# SPLIT DATASET
# ============================
print("=" * 50)
print("MEMULAI PEMBAGIAN DATASET")
print("=" * 50)

total_train = 0
total_test = 0

for plant in os.listdir(RAW_DIR):
    plant_path = RAW_DIR / plant

    if not plant_path.is_dir():
        continue

    for disease in os.listdir(plant_path):
        disease_path = plant_path / disease

        if not disease_path.is_dir():
            continue

        images = [
            f for f in os.listdir(disease_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        split_index = int(len(images) * TRAIN_RATIO)

        train_images = images[:split_index]
        test_images = images[split_index:]

        train_target = TRAIN_DIR / plant / disease
        test_target = TEST_DIR / plant / disease

        train_target.mkdir(parents=True, exist_ok=True)
        test_target.mkdir(parents=True, exist_ok=True)

        # Copy training
        for img in train_images:
            shutil.copy2(
                disease_path / img,
                train_target / img
            )

        # Copy testing
        for img in test_images:
            shutil.copy2(
                disease_path / img,
                test_target / img
            )

        total_train += len(train_images)
        total_test += len(test_images)

        print(
            f"{plant}/{disease}"
            f" -> Training: {len(train_images)}"
            f" | Testing: {len(test_images)}"
        )

print("=" * 50)
print(f"TOTAL TRAINING : {total_train}")
print(f"TOTAL TESTING  : {total_test}")
print("SELESAI!")
print("=" * 50)