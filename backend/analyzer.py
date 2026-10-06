"""
Retina Guard AI - analysis engine
===================================

Two modes, selected automatically:

1. TRAINED MODEL MODE
   If a trained Keras model is found at ``model/retina_model.h5``
   (produced by ``model_training/train_model.py`` on a real dataset such
   as APTOS 2019 or EyePACS, using transfer learning on EfficientNetB0),
   it is loaded and used for inference.

2. DEMO / HEURISTIC MODE (default, works immediately, no dataset needed)
   If no trained model file is present, the analyzer falls back to a
   transparent, rule-based OpenCV pipeline that looks for DR-like visual
   cues (dark lesion spots, bright exudate-like spots) on the green
   channel of the fundus image. This lets the whole app run end-to-end
   out of the box, with no GPU, no internet connection and no dataset.

   IMPORTANT: Heuristic mode is for demonstration / college-project
   purposes only. It is NOT a validated medical model. Swap in the
   trained EfficientNetB0 model (model_training/train_model.py) for
   anything beyond a demo.
"""

import io
import os
import numpy as np
from PIL import Image
import cv2

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "retina_model_weighted.keras"
)

GRADES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

GRADE_COLORS = {
    "No DR": "#2ecc71",
    "Mild": "#f1c40f",
    "Moderate": "#e67e22",
    "Severe": "#e74c3c",
    "Proliferative DR": "#8e0000",
}

GRADE_ADVICE = {
    "No DR": (
        "No signs of diabetic retinopathy were detected by the AI screening model. "
        "This is an educational screening result and does not replace a professional eye examination."
    ),
    "Mild": (
        "Possible retinal abnormalities were detected by the AI screening model. "
        "This result is for educational screening purposes only and should not be considered a medical diagnosis. "
        "Please consult a qualified eye-care professional for appropriate evaluation."
    ),
    "Moderate": (
        "Possible retinal abnormalities were detected by the AI screening model. "
        "This result is for educational screening purposes only and should not be considered a medical diagnosis. "
        "Please consult a qualified eye-care professional for appropriate evaluation."
    ),
    "Severe": (
        "Possible retinal abnormalities were detected by the AI screening model. "
        "This result is for educational screening purposes only and should not be considered a medical diagnosis. "
        "Please consult a qualified eye-care professional for appropriate evaluation."
    ),
    "Proliferative DR": (
        "Possible advanced retinal abnormalities were detected by the AI screening model. "
        "This result is for educational screening purposes only and should not be considered a medical diagnosis. "
        "Please consult a qualified eye-care professional for appropriate evaluation."
    ),
}

_keras_model = None
_keras_available = False

try:
    import tensorflow as tf  # noqa: F401
    _keras_available = True
except ImportError:
    _keras_available = False


def _try_load_trained_model():
    """Load the trained EfficientNetB0 Keras model if available."""
    global _keras_model

    if not _keras_available:
        print("[model] TensorFlow is not available.")
        return None

    if _keras_model is not None:
        return _keras_model

    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow import keras

            print(f"[model] Loading trained model: {MODEL_PATH}")

            _keras_model = keras.models.load_model(
                MODEL_PATH
            )

            print("[model] Trained model loaded successfully.")

            return _keras_model

        except Exception as exc:
            print(f"[model] Could not load trained model: {exc}")
            return None

    print(f"[model] Model not found: {MODEL_PATH}")
    return None


def _read_image(image_bytes: bytes) -> np.ndarray:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_img)


def _is_fundus_like(img_rgb: np.ndarray) -> bool:
    """Rough sanity check: fundus photos are roughly circular with a
    warm (red/orange) dominant hue. Used only to warn the user if they
    upload something that is clearly not a retina photo."""
    h, w, _ = img_rgb.shape
    if h < 50 or w < 50:
        return False
    mean_r = img_rgb[:, :, 0].mean()
    mean_g = img_rgb[:, :, 1].mean()
    mean_b = img_rgb[:, :, 2].mean()
    return mean_r > mean_g and mean_r > mean_b * 0.9


def _heuristic_analyze(img_rgb: np.ndarray):
    """
    Classic image-processing pipeline (demo mode):
      1. Isolate the green channel (lesions show best contrast there).
      2. CLAHE contrast enhancement.
      3. Detect dark blobs (hemorrhage / microaneurysm candidates).
      4. Detect bright blobs (exudate candidates).
      5. Combine lesion counts + area into a severity score 0-4.
      6. Build a heatmap overlay highlighting the detected regions.
    """
    img_resized = cv2.resize(img_rgb, (512, 512))
    green = img_resized[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)

    # Dark lesion candidates (hemorrhages / microaneurysms).
    dark_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 15
    )

    # Bright lesion candidates (exudates) - only the brightest highlights.
    _, bright_thresh = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_or(dark_thresh, bright_thresh)

    # Remove tiny noise specks and thin vessel fragments
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lesion_boxes = []
    total_lesion_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 8 <= area <= 4000:
            x, y, w, h = cv2.boundingRect(c)
            lesion_boxes.append((x, y, w, h))
            total_lesion_area += area

    lesion_count = len(lesion_boxes)
    image_area = 512 * 512
    lesion_density = total_lesion_area / image_area

    # Map lesion_count + density to a 0-4 severity grade.
    # Tuned heuristically for demo purposes, not clinically validated.
    score = lesion_count * 0.6 + lesion_density * 4000
    if score < 8:
        grade_idx = 0
    elif score < 20:
        grade_idx = 1
    elif score < 40:
        grade_idx = 2
    elif score < 70:
        grade_idx = 3
    else:
        grade_idx = 4

    grade_idx = int(np.clip(grade_idx, 0, 4))
    grade = GRADES[grade_idx]

    confidence = float(np.clip(55 + min(lesion_count, 20) * 1.8, 55, 97))

    overlay = img_resized.copy()
    heat_layer = np.zeros_like(img_resized)
    for (x, y, w, h) in lesion_boxes:
        cx, cy = x + w // 2, y + h // 2
        radius = max(w, h) // 2 + 6
        cv2.circle(heat_layer, (cx, cy), radius, (255, 60, 0), -1)

    heat_layer = cv2.GaussianBlur(heat_layer, (25, 25), 0)
    heatmap_overlay = cv2.addWeighted(overlay, 0.75, heat_layer, 0.55, 0)

    return {
        "grade": grade,
        "grade_index": grade_idx,
        "confidence": round(confidence, 1),
        "lesion_count": lesion_count,
        "lesion_density_pct": round(lesion_density * 100, 2),
        "heatmap_image": heatmap_overlay,
        "mode": "heuristic-demo",
    }


def _trained_model_analyze(model, img_rgb: np.ndarray):
    """
    Inference path for a real trained EfficientNetB0 Keras model.
    See model_training/train_model.py for how to produce retina_model.h5
    on a real dataset (APTOS / EyePACS / Messidor / IDRiD). Preprocessing
    here must match exactly what train_model.py used.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img_resized = cv2.resize(img_rgb, (224, 224))
    batch = np.expand_dims(img_resized.astype("float32"), axis=0)
    batch = preprocess_input(batch)

    probs = model.predict(batch, verbose=0)[0]
    grade_idx = int(np.argmax(probs))
    confidence = float(round(probs[grade_idx] * 100, 1))

    overlay_base = cv2.resize(img_rgb, (512, 512))

    # Real explainability: Grad-CAM shows which regions drove the
    # prediction, instead of the OpenCV blob heatmap used in demo mode.
    try:
        from grad_cam import make_gradcam_heatmap, overlay_heatmap
        heatmap = make_gradcam_heatmap(batch, model, pred_index=grade_idx)
        overlay = overlay_heatmap(overlay_base, heatmap)
    except Exception as exc:
        print(f"[grad_cam] falling back to plain image ({exc})")
        overlay = overlay_base

    return {
        "grade": GRADES[grade_idx],
        "grade_index": grade_idx,
        "confidence": confidence,
        "lesion_count": None,
        "lesion_density_pct": None,
        "heatmap_image": overlay,
        "mode": "trained-model",
    }


def analyze_image(image_bytes: bytes):
    img_rgb = _read_image(image_bytes)
    fundus_like = _is_fundus_like(img_rgb)

    model = _try_load_trained_model()
    if model is not None:
        result = _trained_model_analyze(model, img_rgb)
    else:
        result = _heuristic_analyze(img_rgb)

    result["fundus_like"] = bool(fundus_like)
    result["color"] = GRADE_COLORS[result["grade"]]
    result["advice"] = GRADE_ADVICE[result["grade"]]
    return result


def encode_image_to_bytes(img_rgb_array: np.ndarray) -> bytes:
    img_bgr = cv2.cvtColor(img_rgb_array, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise RuntimeError("Could not encode heatmap image")
    return buf.tobytes()
