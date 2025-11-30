from pathlib import Path
import cv2 as cv
from ultralytics import YOLO
from video_analyzing_scripts.feature_engineering import handle_display_features
from video_analyzing_scripts.rim import handle_detect_net_moved, handle_detetect_rim, handle_draw_net_moved
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point, handle_draw_trajectory

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v6/weights/best.pt"
video_path = project_dir / "data/collection/trafione/scored8.mp4"

model = YOLO(model_path)

basketball_class_index = 0
rim_class_index = 0

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)

trajectory_points = []
detected_rim_box = None #[xmin, ymin, xmax, ymax]
frame_count = 0
prev_frame = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Film nie posiada więcej klatek do wyświetlenia")
        break

    frame_count += 1
    boxes = model(frame)[0].boxes

    for box in boxes:
        point = handle_detect_trajectory_point(frame_count, box)
        detected_rim_box = handle_detetect_rim(frame, box)
        net_attr = None

        if prev_frame is not None:
            net_attr = handle_detect_net_moved(frame, prev_frame, box)
            
        if point is not None:
            trajectory_points.append(point)

        if net_attr is not None:
            handle_draw_net_moved(frame, net_attr[0], net_attr[1])
        
    handle_draw_trajectory(frame, trajectory_points)

    cv.imshow("Analiza rzutu", frame)

    prev_frame = frame.copy()

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
handle_display_features(trajectory_points, detected_rim_box, fps)

