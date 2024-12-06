from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="./cfg/zhy.yaml", epochs=150, imgsz=640)
