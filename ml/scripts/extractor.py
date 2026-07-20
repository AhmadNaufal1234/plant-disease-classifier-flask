import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def segment_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 30, 20])
    upper = np.array([100, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if cv2.countNonZero(mask) < 100:
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

    return mask


def extract_lesion_ratio(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    leaf_pixels = v_channel[mask > 0]
    if len(leaf_pixels) < 10:
        return 0.0

    _, thresh = cv2.threshold(
        leaf_pixels.astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    lesion_ratio = np.sum(thresh == 0) / len(leaf_pixels)
    return lesion_ratio


def extract_hue_histogram(image, mask, bins=6):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-8)
    return hist.tolist()


def extract_features(image):
    image = cv2.resize(image, (160, 160))
    mask = segment_leaf(image)

    # ===== HSV: mean + std =====
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0][mask > 0]
    s_channel = hsv[:, :, 1][mask > 0]
    v_channel = hsv[:, :, 2][mask > 0]

    h_mean, h_std = np.mean(h_channel), np.std(h_channel)
    s_mean, s_std = np.mean(s_channel), np.std(s_channel)
    v_mean, v_std = np.mean(v_channel), np.std(v_channel)

    # ===== GLCM =====
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    bins = 32
    gray_quantized = (gray_masked.astype(np.float64) / 256 * bins).astype(np.uint8)
    gray_quantized = np.clip(gray_quantized, 0, bins - 1)

    glcm = graycomatrix(
        gray_quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=bins,
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, "contrast").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    energy = graycoprops(glcm, "energy").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    asm = graycoprops(glcm, "ASM").mean()

    # ===== Fitur bercak & warna =====
    lesion_ratio = extract_lesion_ratio(image, mask)
    hue_hist = extract_hue_histogram(image, mask, bins=6)

    return [
        h_mean, s_mean, v_mean,
        h_std, s_std, v_std,
        contrast, correlation, energy,
        homogeneity, dissimilarity, asm,
        lesion_ratio,
        *hue_hist,
    ]


FEATURE_COLUMNS = [
    "h_mean", "s_mean", "v_mean",
    "h_std", "s_std", "v_std",
    "contrast", "correlation", "energy",
    "homogeneity", "dissimilarity", "asm",
    "lesion_ratio",
    "hue_hist_0", "hue_hist_1", "hue_hist_2",
    "hue_hist_3", "hue_hist_4", "hue_hist_5",
]