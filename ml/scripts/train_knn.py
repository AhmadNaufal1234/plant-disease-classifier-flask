import pandas as pd
import joblib
import numpy as np

from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from extractor import FEATURE_COLUMNS

# ==========================
# PATH
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "features" / "train_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================
# LOAD DATA
# ==========================
print("=" * 50)
print("MEMUAT DATA TRAINING")
print("=" * 50)

df = pd.read_csv(FEATURE_FILE)

X = df[FEATURE_COLUMNS]
y = df["plant"] + "_" + df["label"]

print(f"Jumlah Data  : {len(df)}")
print(f"Jumlah Fitur : {X.shape[1]}")

# Cek distribusi kelas — penting! Kalau timpang, itu bisa jadi penyebab akurasi rendah
print("\nDistribusi Kelas:")
print(y.value_counts())

# ==========================
# NORMALISASI
# ==========================
print("=" * 50)
print("NORMALISASI FITUR")
print("=" * 50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# CARI K TERBAIK OTOMATIS
# ==========================
print("=" * 50)
print("MENCARI NILAI K TERBAIK (Cross-Validation)")
print("=" * 50)

param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid.fit(X_scaled, y)

print(f"Parameter Terbaik : {grid.best_params_}")
print(f"Akurasi CV Terbaik: {grid.best_score_:.4f}")

model = grid.best_estimator_

# ==========================
# SIMPAN MODEL
# ==========================
model_path = MODEL_DIR / "knn_model.pkl"
scaler_path = MODEL_DIR / "scaler.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("=" * 50)
print("MODEL BERHASIL DISIMPAN")
print(f"Model  : {model_path}")
print(f"Scaler : {scaler_path}")
print("=" * 50)