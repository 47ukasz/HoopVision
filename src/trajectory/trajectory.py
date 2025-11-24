from pathlib import Path
import cv2 as cv
import numpy as np
from src.trajectory import feature_engineering as fe
from ultralytics import YOLO

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v6/weights/best.pt"
video_path = project_dir / "data/collection/videos/nietrafione/missed12.mp4"
#missed11

model = YOLO(model_path)
basketball_class_index = 0
rim_class_index = 1

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)

trajectory_points = []
detected_rim_box = None #[xmin, ymin, xmax, ymax]
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Film nie posiada więcej klatek do wyświetlenia...")
        break

    frame_count += 1
    boxes = model(frame)[0].boxes

    for box in boxes:
        box_class_index = int(box.cls[0])
        box_conf_level = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()

        if box_class_index == basketball_class_index and box_conf_level > 0.5:
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)

            point = (frame_count, center_x, center_y)
            trajectory_points.append(point)
        
        elif box_class_index == rim_class_index and box_conf_level > 0.3:
            detected_rim_box = xyxy
    
    if detected_rim_box is not None:
        rx1, ry1, rx2, ry2 = map(int, detected_rim_box)
        cv.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        
        target_x = int((rx1 + rx2) / 2)
        target_y = int((ry1 + ry2) / 2) 
        cv.circle(frame, (target_x, target_y), 3, (0, 255, 255), -1)   
        #krawedzie tunelu
        rim_width = rx2 - rx1
        margin = int(rim_width * 0.15) 
        cv.line(frame, (rx1 - margin, ry2 - margin), (rx1, ry2 + 200), (0, 255, 0), 2)
        cv.line(frame, (rx2 + margin, ry2 + margin), (rx2, ry2 + 200), (0, 255, 0), 2)

    for index in range(1, len(trajectory_points)):
        start_point = trajectory_points[index-1][1:] 
        end_point = trajectory_points[index][1:]
        cv.line(frame, start_point, end_point, (255, 0, 0), 5)
    
    cv.imshow("Trajektoria", frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

if detected_rim_box is not None and len(trajectory_points) > 5:
    wyniki = fe.feature_extraction(trajectory_points, detected_rim_box, fps)
    
    print(f"Minimalna odległość od centrum: {wyniki['min_odleglosc_pix']} px")
    print(f"Prędkość: {wyniki['predkosc']} px/frame")
    print(f"Kąt: {wyniki['kat']} stopni")
    print(f"Czy skończyło się pod obręczą?: {'TAK' if wyniki['czy_w_tunelu_pod_obrecza'] else 'NIE'}")
    
else:
    print("Nie udało się zebrać wystarczających danych (brak obręczy lub krótka trajektoria).")