from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v1/weights/best.pt"
video_path = project_dir / "data/collection/videos/video2.mp4"

model = YOLO(model_path)
basketball_class_index = 0

cap = cv.VideoCapture(video_path)

trajectory_points = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Film nie posiada więcej klatek do wyświetlenia...")
        break

    boxes = model(frame)[0].boxes

    for box in boxes:
        box_class_index = int(box.cls[0])
        box_conf_level = float(box.conf[0])

        if box_class_index == basketball_class_index and box_conf_level > 0.4:
            center_x, center_y = box.xywh[0][:2]

            point = (int(center_x), int(center_y))
            trajectory_points.append(point)
            break            

    for index in range(1, len(trajectory_points)):
        cv.line(frame, trajectory_points[index-1], trajectory_points[index], (255,0,0), 5)
    
    cv.imshow("Trajektoria", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()