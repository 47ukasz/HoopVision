from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v1/weights/best.pt"
asset_path = project_dir / "data/collection/videos/video4.mp4"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
unique_name = f"{asset_path.stem}_{timestamp}"

model = YOLO(model_path)

results = model.predict(
    source=asset_path,
    device="mps", # to dla maca jak coś, na windowsie/linux'ie chyba trzeba zmienic
    conf=0.25,
    iou=0.7,
    imgsz=640,
    save=True,        
    save_txt=True,
    project="predictions",
    name=unique_name,   
    vid_stride=1
)