from pathlib import Path
import csv
import math
import sys
import joblib
import pandas as pd

import cv2 as cv
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from video_analyzing_scripts.trajectory import handle_detect_trajectory_point, handle_draw_trajectory
from video_analyzing_scripts.feature_engineering import feature_extraction, handle_display_features
from video_analyzing_scripts.rim import handle_detect_net_moved, handle_detetect_rim, handle_draw_net_moved
from constants import BASKETBALL_CLASS_INDEX, MIN_BALL_CONF, RIM_CLASS_INDEX, MIN_RIM_CONF

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v6/weights/best.pt"
videos_path = project_dir / "data/collection/all"
video_path = project_dir / "data/collection/all/scored8.mp4"
csv_path = project_dir / "data/trajectory_features.csv"
knn_model_path = project_dir / "models/knn_model.pkl"

model = YOLO(model_path)
knn_model = joblib.load(knn_model_path)

def analyze_video(video_path):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)

    trajectory_points = []
    detected_rim_box = None #[xmin, ymin, xmax, ymax]
    frame_count = 0
    prev_frame = None
    point = None
    net_moved = False

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Film nie posiada więcej klatek do wyświetlenia")
            break

        frame_count += 1
        boxes = model(frame)[0].boxes

        ballSelected = False
        rimSelected = False

        for box in boxes:
            box_class_index = int(box.cls[0])
            box_conf_level = float(box.conf[0])

            if box_class_index == BASKETBALL_CLASS_INDEX and box_conf_level >= MIN_BALL_CONF and not ballSelected:
                point = handle_detect_trajectory_point(frame_count, box)
                ballSelected = True

            if box_class_index == RIM_CLASS_INDEX and box_conf_level >= MIN_RIM_CONF and not rimSelected:
                detected_rim_box = handle_detetect_rim(frame, box)
                rimSelected = True

        net_attr = None
            
        if point is not None:
            trajectory_points.append(point)
        
        distance = handle_calculate_distance(detected_rim_box, point)

        if prev_frame is not None and distance is not None and distance <= 250:
            net_attr = handle_detect_net_moved(frame, prev_frame, box)
        
        if net_attr is not None and net_attr[0]:
            net_moved = True

        prev_frame = frame.copy()

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    handle_display_features(trajectory_points, detected_rim_box, fps)

    result = feature_extraction(trajectory_points, detected_rim_box, fps)

    features = feature_extraction(trajectory_points, detected_rim_box, fps)
    features["net_moved"] = 1 if net_moved else 0
                        
    features_df = pd.DataFrame([features])
    prediction = knn_model.predict(features_df)
    print("Czy trafiono?:", 'TAK' if prediction[0] else 'Nie')  # 0 – pudło, 1 – trafiony

    return {
        "file_name": video_path.name,
        "min_odleglosc_pix": result.get("min_odleglosc_pix"),
        "predkosc": result.get("predkosc"),
        "kat": result.get("kat"),
        "czy_w_tunelu_pod_obrecza": 1 if result.get("czy_w_tunelu_pod_obrecza") else 0,
        "net_moved": 1 if net_moved else 0
    }

def handle_calculate_distance(rimCords, ballCords):
    if rimCords is None or ballCords is None:
        return
    
    rx1, ry1, rx2, ry2 = rimCords

    rim_center_x = (rx1 + rx2) / 2
    rim_center_y = (ry1 + ry2) / 2

    _, ball_center_x, ball_center_y = ballCords;

    distance = math.sqrt(math.pow((ball_center_x - rim_center_x),2) + math.pow((ball_center_y - rim_center_y), 2))
        
    return distance

def save_csv(rows, csv_path):
    if not rows:
        print("Brak danych do zapisania.")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Zapisano wyniki do: {csv_path}")

videos = list(videos_path.glob("*.mp4"))
results = []

for video_path in videos:
    res = analyze_video(video_path)
    if res is not None:
        results.append(res)

save_csv(results, csv_path)

