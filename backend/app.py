"""
Retina Guard AI - Flask backend
=================================
Plain Flask backend serving the HTML/CSS/Bootstrap/JS frontend and a
JSON prediction API.

Run with:
    python app.py

Then open http://localhost:5000 in a browser.
"""

import base64
import io
import os

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from analyzer import analyze_image, encode_image_to_bytes
from report import build_report_pdf
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
SAMPLES_DIR = os.path.join(BASE_DIR, "..", "sample_images")

app = Flask(__name__, static_folder=None)
CORS(app)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE_MB = 15
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


@app.route("/")
def serve_home():
    return send_from_directory(FRONTEND_DIR, "home.html")


@app.route("/screening")
def serve_screening():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/history")
def serve_history_page():
    return send_from_directory(FRONTEND_DIR, "history.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/api/samples")
def list_samples():
    """Return the list of bundled demo fundus images for the gallery."""
    files = sorted(
        f for f in os.listdir(SAMPLES_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    )
    return jsonify({"samples": [f"/samples/{f}" for f in files]})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Please upload a JPG, PNG or BMP image."}), 400

    image_bytes = file.read()

    try:
        result = analyze_image(image_bytes)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 400

    heatmap_bytes = encode_image_to_bytes(result.pop("heatmap_image"))
    heatmap_b64 = base64.b64encode(heatmap_bytes).decode("utf-8")

    # Log to MySQL if configured; silently skipped otherwise (see db.py)
    db.save_prediction(
        filename=file.filename,
        grade=result["grade"],
        confidence=result["confidence"],
        lesion_count=result["lesion_count"],
        mode=result["mode"],
    )

    return jsonify({
        "grade": result["grade"],
        "grade_index": result["grade_index"],
        "confidence": result["confidence"],
        "color": result["color"],
        "advice": result["advice"],
        "lesion_count": result["lesion_count"],
        "lesion_density_pct": result["lesion_density_pct"],
        "fundus_like": result["fundus_like"],
        "mode": result["mode"],
        "heatmap_image_base64": f"data:image/jpeg;base64,{heatmap_b64}",
    })


@app.route("/api/report", methods=["POST"])
def report():
    """Re-analyzes the uploaded image and returns a one-page PDF report.
    Re-running analysis (rather than caching) keeps the server stateless."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Please upload a JPG, PNG or BMP image."}), 400

    image_bytes = file.read()

    try:
        result = analyze_image(image_bytes)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 400

    heatmap_jpeg_bytes = encode_image_to_bytes(result.pop("heatmap_image"))

    pdf_bytes = build_report_pdf(
        heatmap_jpeg_bytes=heatmap_jpeg_bytes,
        grade=result["grade"],
        confidence=result["confidence"],
        lesion_count=result["lesion_count"],
        lesion_density_pct=result["lesion_density_pct"],
        advice=result["advice"],
        mode=result["mode"],
        filename=file.filename,
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="retina_guard_report.pdf",
    )


@app.route("/api/history")
def history():
    """Returns recent predictions from MySQL (empty list if DB not set up)."""
    return jsonify({"history": db.get_history()})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
