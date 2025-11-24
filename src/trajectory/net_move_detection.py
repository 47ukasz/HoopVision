from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v1/weights/best.pt"
video_path = project_dir / "data/collection/videos/train/video.mp4"

model = YOLO(model_path)
rim_class_index = 2

cap = cv.VideoCapture(video_path)
prev_gray_scale_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Film nie posiada więcej klatek do wyświetlenia...")
        break

    boxes = model(frame)[0].boxes

    gray_scale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 

    for box in boxes:
        box_class_index = int(box.cls[0])
        box_conf_level = float(box.conf[0])

        if box_class_index == rim_class_index and box_conf_level > 0.2:
            rim_moving = False

            if prev_gray_scale_frame is None:
                break
            
            cords = [int(cord) for cord in box.xyxy[0]]
            x1, y1, x2, y2 = cords

            prev_area = prev_gray_scale_frame[y1:y2, x1:x2]
            curr_area = gray_scale_frame[y1:y2, x1:x2]

            if prev_area.size > 0 and curr_area.size > 0:
                flow = cv.calcOpticalFlowFarneback(prev_area, curr_area, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 7, poly_sigma = 1.1, flags = 0)
                mag, ang = cv.cartToPolar(flow[:,:,0], flow[:,:,1])

                rim_motion_value = np.mean(mag)
                rim_moving = rim_motion_value > 2.

                label = f"{"Nie" if not rim_moving else ""} porusza sie"
                label_color = (255, 0, 0) if not rim_moving else (0, 0, 255)
                cv.putText(frame, label, (x1, y1 - 20), cv.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            
            rim_color = (255, 0, 0) if not rim_moving else (0, 0, 255)

            cv.rectangle(frame, (x1, y1), (x2, y2), rim_color, 2)

    cv.imshow("Detekcja siatki", frame)

    prev_gray_scale_frame = gray_scale_frame.copy()

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()