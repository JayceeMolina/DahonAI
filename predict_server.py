from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

app = Flask(__name__)
CORS(app)

# Folder paths
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
MODEL_PATH = "models/last.pt"
model = YOLO(MODEL_PATH)

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded image temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    time.sleep(0.2)

    try:
        # Read image
        image_data = np.fromfile(filepath, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not read image"}), 400

        # Run YOLO prediction
        result = model.predict(source=image, save=False, show=False)[0]

        # Create segmented output
        segmented_image = result.plot()

        # Save output images
        cv2.imwrite(os.path.join(RESULTS_FOLDER, "orig_" + file.filename), image)
        cv2.imwrite(os.path.join(RESULTS_FOLDER, "seg_" + file.filename), segmented_image)

        # Store detections
        detections = []

        if len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                class_id = int(result.boxes.cls[i])
                confidence = float(result.boxes.conf[i])

                detections.append({
                    "class_name": model.names[class_id],
                    "confidence": round(confidence, 4),
                    "class_id": class_id
                })

        elif hasattr(result, "probs") and result.probs is not None:
            class_id = result.probs.top1
            confidence = float(result.probs.top1conf)

            detections.append({
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4),
                "class_id": class_id
            })

        else:
            detections.append({
                "class_name": "Unknown",
                "confidence": 0.0,
                "class_id": -1
            })

        # Image URLs
        original_url = f"http://127.0.0.1:5000/results/orig_{file.filename}"
        segmented_url = f"http://127.0.0.1:5000/results/seg_{file.filename}"

        return jsonify({
            "detections": detections,
            "total_detections": len(detections),
            "orig_image": original_url,
            "segmented_image": segmented_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Remove temporary upload
        if os.path.exists(filepath):
            os.remove(filepath)

# Serve result images
@app.route("/results/<filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)