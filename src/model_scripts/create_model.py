from ultralytics import YOLO
from utils.model import create_next_model_folder

model = YOLO("yolo11s.pt")

model_folder = create_next_model_folder()
model_name = model_folder.name

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    device="mps", # to dla maca jak coś, na windowsie/linux'ie chyba trzeba zmienic             
    project="models",
    name="model_name",
    name="run1"
)
