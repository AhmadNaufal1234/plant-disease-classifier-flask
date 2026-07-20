from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import joblib
import numpy as np
import sys

from pathlib import Path

app = Flask(__name__)
CORS(app)

# ==========================
# PATH SETUP
# ==========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "ml" / "scripts"))

from ml.core.evaluation import run_evaluation
from extractor import extract_features  # fungsi ekstraksi yang SAMA dengan training

# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "knn_model.pkl"
SCALER_PATH = PROJECT_ROOT / "ml" / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==========================
# PREDICT
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "image not found"}), 400

        file = request.files["image"]

        image_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "invalid image"}), 400

        # extract_features() sudah handle resize (160x160) + segmentasi daun
        # di dalamnya sendiri — TIDAK perlu resize manual di sini lagi
        features = np.array([extract_features(image)])

        features = scaler.transform(features)

        # ==========================
        # CEK JARAK KE TETANGGA TERDEKAT
        # Kalau gambar terlalu "asing" dibanding data training
        # (misal bukan daun sama sekali), tolak prediksi.
        # ==========================
        distances, _ = model.kneighbors(features)
        avg_distance = float(np.mean(distances))

        DISTANCE_THRESHOLD = 40.0  # TODO: sesuaikan berdasarkan hasil testing

        if avg_distance > DISTANCE_THRESHOLD:
            return jsonify({
                "error": "unrecognized",
                "message": "Gambar tidak dikenali sebagai daun tanaman. Pastikan foto menampilkan daun dengan jelas.",
                "avg_distance": round(avg_distance, 2)
            }), 200

        # Predict
        prediction = model.predict(features)[0]

        # Confidence
        probabilities = model.predict_proba(features)
        confidence = float(np.max(probabilities) * 100)

        # Split label
        parts = prediction.split("_", 1)
        plant = parts[0]
        disease = parts[1] if len(parts) > 1 else "unknown"

        return jsonify({
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# HOME
# ==========================
@app.route("/")
def home():
    return jsonify({"message": "AgroScan AI API Running"})


# ==========================
# EVALUATE MODEL
# ==========================
@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data = request.get_json()
        k = int(data["k"])
        split = data["split"]

        csv_path = PROJECT_ROOT / "ml" / "features" / "train_features.csv"

        result = run_evaluation(csv_path=csv_path, k=k, split=split)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)