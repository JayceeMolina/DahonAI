from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("models/yolo11n-seg.pt") # Your .pt model path

# Train model using dataset configuration
results = model.train(data="dataset/data.yaml", epochs=100, imgsz=640)# Dataset YAML file, Number of training epochs, Input image size