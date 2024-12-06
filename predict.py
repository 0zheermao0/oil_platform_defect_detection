import os
import torch
from ultralytics import YOLO
from typing import List, Optional

model_path = "./cfg/best.pt"
yolo_model = YOLO(model_path)

def process_frame():
    pass

def process_images(model_path: str, source_dir: str, output_dir: str, imgsz: int=640, conf: float=0.5, classes: list=None, name: str='predict') -> List[str]:
    """
    Processes images in the source directory using a YOLOv8 model and saves the results to the output directory.

    Parameters:
    - model: the pre-trained YOLO model.
    - source_dir (str): Directory containing images to process.
    - output_dir (str): Directory where processed results will be saved.
    - imgsz (int, optional): Image size for inference. Default is 640.
    - conf (float, optional): Confidence threshold for detections. Default is 0.5.
    - classes (list, optional): List of class indices to detect. Default is None (detect all classes).
    - device (str, optional): Device to run inference on ('cpu', 'cuda', etc.). Default is 'cpu'.
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = yolo_model
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Run inference on the source directory
        results = model.predict(source=source_dir, save=True, project=output_dir, name=name, exist_ok=True, imgsz=imgsz, conf=conf, classes=classes, device=device)
        # return all processed image paths
        # print(f'return images paths: {[os.path.join(output_dir, name, f) for f in os.listdir(os.path.join(output_dir, name)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]}')
        return [os.path.join(output_dir, name, f) for f in os.listdir(os.path.join(output_dir, name)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# model_path = "best.pt"
# source_dir = "/root/exp/zhy/datasets/label-studio/media/upload/2"
# output_dir = "datasets/out_images"
# process_images(model_path, source_dir, output_dir, device='cuda')

