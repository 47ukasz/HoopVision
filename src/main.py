from pathlib import Path
import pandas as pd
import joblib
import cv2 as cv
from ultralytics import YOLO
from video_analyzing_scripts.feature_engineering import handle_display_features, feature_extraction
from video_analyzing_scripts.rim import handle_detect_net_moved, handle_detetect_rim, handle_draw_net_moved
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point, handle_draw_trajectory
from video_analyzing_scripts.throw_roi import create_tracking_roi, is_ball_in_roi
from constants import RIM_CLASS_INDEX, MIN_RIM_CONF, BASKETBALL_CLASS_INDEX, MIN_BALL_CONF
project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v6/weights/best.pt"
knn_model_path = project_dir / "models/knn_model.pkl"
video_path = project_dir / "data/collection/trafione/scored1.mp4"

model = YOLO(model_path)
knn_model = joblib.load(knn_model_path)

basketball_class_index = 0
rim_class_index = 0

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)

trajectory_points = []
detected_rim_box = None #[xmin, ymin, xmax, ymax]
rim_locked = False
frame_count = 0
prev_frame = None
point = None

## zmienne do logiki rzutow
shot_cooldown = 0            #licznik przerwy miedzy rzutami
COOLDOWN_DURATION = int(fps * 2) #przerwa miedzy sledzeniem
is_tracking_shot = False     #czy sledzenie jest wlaczone
tracking_roi = None          # obszar region of intrest
total_shots = 0

shot_net_moved = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Film nie posiada więcej klatek do wyświetlenia")
        break

    frame_count += 1
    
    # licznik rzutow
    cv.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
    cv.putText(frame, f"Rzuty: {total_shots}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # logika przerwy sledzenia
    if shot_cooldown > 0:
        shot_cooldown -= 1
        cv.putText(frame, f"Cooldown: {shot_cooldown}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
        # rysuje strefe na szaro(nieaktywne)
        if tracking_roi:
            cv.rectangle(frame, (tracking_roi[0], tracking_roi[1]), (tracking_roi[2], tracking_roi[3]), (100, 100, 100), 2)
        
        cv.imshow("Analiza rzutu", frame)
        if cv.waitKey(1) == ord('q'): break
        continue

    # detekcja
    results = model(frame, verbose=False, imgsz=640)
    boxes = results[0].boxes

    net_attr = None
    current_ball_box = None

    # iterujemy po wszyskich klasach
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # 1. RIM
        if cls_id == RIM_CLASS_INDEX and conf >= MIN_RIM_CONF:
            detected_rim_box = handle_detetect_rim(frame, box)
            # Ustawienie ROI
            tracking_roi = create_tracking_roi(detected_rim_box, frame.shape)
            
            # detekcja ruchu siatki 
            net_attr = handle_detect_net_moved(frame, prev_frame, box)
                
            if is_tracking_shot and net_attr is not None:
                is_moving, _ = net_attr
                
                if is_moving:
                    shot_net_moved = True

        # 2. BASKETBALL
        elif cls_id == BASKETBALL_CLASS_INDEX and conf >= MIN_BALL_CONF:
            point = handle_detect_trajectory_point(frame_count, box)
            
            if point is not None:
                current_ball_box = box.xyxy[0].cpu().numpy()
                
                # sprawdzamy czy pilka w ROI
                in_zone = tracking_roi and is_ball_in_roi(current_ball_box, tracking_roi)

                if in_zone:
                    if not is_tracking_shot:
                        is_tracking_shot = True
                        trajectory_points = [] # reset dla nowego rzutu
                    trajectory_points.append(point)
                else:
                    # Sprawdzenie czy pilka poza strefa -> koniec rzutu
                    if is_tracking_shot:
                        total_shots += 1
                        print(f"Rzut nr {total_shots} zakończony. Analiza...")
                        
                        features = feature_extraction(trajectory_points, detected_rim_box, fps)
                        features["net_moved"] = shot_net_moved
                        
                        features_df = pd.DataFrame([features])
                        prediction = knn_model.predict(features_df)
                        print("Czy trafiono?:", 'TAK' if prediction[0] else 'Nie')  # 0 – pudło, 1 – trafiony
                        # Reset
                        is_tracking_shot = False
                        shot_net_moved = False
                        shot_cooldown = COOLDOWN_DURATION
                        trajectory_points = []

    # rysowanie ROI
    if tracking_roi:
        color = (0, 255, 0) if is_tracking_shot else (255, 0, 0) # Zielony jak nagrywa, niebieski jak czuwa
        cv.rectangle(frame, (tracking_roi[0], tracking_roi[1]), (tracking_roi[2], tracking_roi[3]), color, 2)
        cv.putText(frame, "ROI", (tracking_roi[0], tracking_roi[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if net_attr is not None:
        handle_draw_net_moved(frame, net_attr[0], net_attr[1])
        
    if is_tracking_shot:
        handle_draw_trajectory(frame, trajectory_points)

    cv.imshow("Analiza rzutu", frame)
    prev_frame = frame.copy()

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()