import os
import sys
import cv2
import pandas as pd

from pathlib import Path

# ==========================
# PATH
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "training"
FEATURE_DIR = BASE_DIR / "features"

FEATURE_DIR.mkdir(exist_ok=True)

# Import fungsi ekstraksi yang SAMA dipakai training & evaluasi
from extractor import extract_features, FEATURE_COLUMNS

# ==========================
# EKSTRAKSI
# ==========================
rows = []

print("=" * 50)
print("EKSTRAKSI FITUR TRAINING")
print("=" * 50)

for plant in os.listdir(TRAIN_DIR):
    plant_path = TRAIN_DIR / plant

    if not plant_path.is_dir():
        continue

    for disease in os.listdir(plant_path):
        disease_path = plant_path / disease

        if not disease_path.is_dir():
            continue

        for file_name in os.listdir(disease_path):
            image_path = disease_path / file_name

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            # extract_features() sudah handle resize di dalamnya
            feature_vector = extract_features(image)

            row = feature_vector + [plant, disease]
            rows.append(row)

        print(f"✔ {plant}/{disease}")

# ==========================
# SIMPAN CSV
# ==========================
columns = FEATURE_COLUMNS + ["plant", "label"]

df = pd.DataFrame(rows, columns=columns)

output_path = FEATURE_DIR / "train_features.csv"
df.to_csv(output_path, index=False)

print("=" * 50)
print("Jumlah Data :", len(df))
print("Jumlah Fitur :", len(FEATURE_COLUMNS))
print("File Tersimpan :", output_path)
print("=" * 50)