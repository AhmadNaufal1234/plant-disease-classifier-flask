import os
import cv2
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score
)

# ==========================
# PATH
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    BASE_DIR
    / "features"
    / "train_features.csv"
)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(FEATURE_FILE)

X = df[
    [
        "h_mean",
        "s_mean",
        "v_mean",
        "contrast",
        "correlation",
        "energy",
        "homogeneity",
    ]
]

y = (
    df["plant"]
    + "_"
    + df["label"]
)

# ==========================
# NORMALISASI
# ==========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================
# TEST K
# ==========================
print("=" * 50)
print("MENCARI NILAI K TERBAIK")
print("=" * 50)

best_k = 0
best_accuracy = 0

for k in range(1, 16):

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )

    model.fit(
        X_scaled,
        y
    )

    y_pred = model.predict(
        X_scaled
    )

    accuracy = accuracy_score(
        y,
        y_pred
    )

    print(
        f"K = {k:<2} | Accuracy = {accuracy:.4f}"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_k = k

print("=" * 50)
print(
    f"K TERBAIK : {best_k}"
)
print(
    f"AKURASI   : {best_accuracy:.4f}"
)
print("=" * 50)