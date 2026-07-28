# 🍃 Dahon Detection System

An AI-powered leaf detection system using **YOLO Instance Segmentation**, **Flask API**, and **OpenCV** for image-based plant analysis.

The system allows users to upload an image, runs YOLO segmentation, detects leaf objects, generates segmented output images, and returns detection results through a REST API.

---

# 📌 Features

- 🌿 YOLO-based leaf detection
- 🎯 Instance segmentation
- 📷 Image upload prediction API
- 🖼️ Original and segmented image output
- ⚡ Flask REST API
- 🔍 Confidence score display
- 📦 Multiple object detection support

---

# 🏗️ System Flow

```
User Upload Image
        |
        v
     Flask API
        |
        v
 YOLO Segmentation Model
        |
        v
 Detection Results
        |
        +------------+
        |            |
        v            v
 Original Image   Segmented Image
```

---

# 🛠️ Technologies Used

## Artificial Intelligence
- Python
- Ultralytics YOLO
- YOLO Instance Segmentation
- OpenCV

## Backend
- Flask
- Flask-CORS

---

# 📂 Project Structure

```
Dahon_Final/
│
├── predict_server.py       # Flask prediction API
├── training.py             # YOLO model training script
├── plot_training.py        # Generate training graphs
├── converter.py            # Dataset/model utility script
│
├── models/
│   └── last.pt             # YOLO model weights (not included)
│
├── uploads/                # Temporary uploaded images
├── results/                # Generated prediction images
│
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/JayceeMolina/Dahon_Final.git

cd Dahon_Final
```

---

## 2. Install Dependencies

```bash
pip install flask
pip install flask-cors
pip install ultralytics
pip install opencv-python
pip install numpy
```

---

# 🤖 YOLO Model Setup

The trained YOLO model is not included in this repository.

Place your trained model here:

```
models/last.pt
```

The API loads the model using:

```python
model = YOLO("models/last.pt")
```

---

# 📚 Dataset Training

The dataset is excluded because of size limitations.

Dataset structure:

```
dataset/
│
├── data.yaml
├── train/
└── valid/
```

Train the model:

```python
from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("yolo11n-seg.pt")

# Train model using dataset configuration
results = model.train(
    data="dataset/data.yaml", # Dataset YAML file
    epochs=100,               # Number of training epochs
    imgsz=640                 # Input image size
)
```

---

# ▶️ Running the API

Start Flask server:

```bash
python predict_server.py
```

Server runs at:

```
http://127.0.0.1:5000
```

---

# 📡 API Usage

## Endpoint

```
POST /predict
```

Upload an image:

```
image = your_image.jpg
```

Example response:

```json
{
    "detections": [
        {
            "class_name": "leaf",
            "confidence": 0.95,
            "class_id": 0
        }
    ],
    "total_detections": 1,
    "orig_image": "results/orig_image.jpg",
    "segmented_image": "results/seg_image.jpg"
}
```

---

# 📊 Training Visualization

After training, generate training graphs:

```bash
python plot_training.py
```

Requires:

```
runs/segment/train/results.csv
```

---

# 🚫 Excluded Files

The following are ignored:

```
*.pt
dataset/
runs/
uploads/
results/
```

These files are excluded because they contain:

- Large AI model weights
- Training datasets
- Temporary generated files

---

# 👨‍💻 Author

Jaycee Molina

Computer Engineer

GitHub:
https://github.com/JayceeMolina
