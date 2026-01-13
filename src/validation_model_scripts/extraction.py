from pathlib import Path
import pandas as pd
import cv2 as cv
from ultralytics import YOLO
import numpy as np

from video_analyzing_scripts.feature_engineering import feature_extraction
from video_analyzing_scripts.rim import (
    handle_detect_net_moved, 
    handle_detect_rim, 
    handle_calculate_ball_in_rim
)
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point
from video_analyzing_scripts.throw_roi import create_tracking_roi, is_ball_in_roi
from constants import RIM_CLASS_INDEX, MIN_RIM_CONF, BASKETBALL_CLASS_INDEX, MIN_BALL_CONF

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v7/weights/best.pt"

input_videos_dir = project_dir / "data/collection/videos/odbite" 
# input_videos_dir = project_dir / "data/collection/videos/nietrafione"

video_files = list(input_videos_dir.glob("*.mp4"))
csv_output_path = project_dir / "extracted_features_final_odbite.csv"

# Model YOLO
model = YOLO(model_path)
all_shots_data = []

# Kolumny wyjściowe
REQUIRED_COLUMNS = [
    "file_name", "label", "min_odleglosc_pix", 
    "predkosc", "kat", "czy_w_tunelu_pod_obrecza", "net_moved"
]

print(f"Znaleziono {len(video_files)} filmów w: {input_videos_dir}")

for video_idx, video_path in enumerate(video_files):
    print(f"--- Przetwarzanie [{video_idx + 1}/{len(video_files)}]: {video_path.name} ---")
    
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Błąd otwarcia pliku: {video_path.name}")
        continue

    fps = cap.get(cv.CAP_PROP_FPS)
    
    shotState = "IDLE"
    trajectory_points = []
    last_ball_xyxy = None
    last_point = None
    frame_count = 0
    prev_frame = None
    
    shot_cooldown = 0
    COOLDOWN_DURATION = int(fps * 2)
    NET_MOVED_DETECTION_DURATION = int(fps * 0.2)
    
    tracking_roi = None
    
    ball_in_rim_prev = False
    net_moved_detection_cooldown = 0
    shot_net_moved = False
    net_moved_in_frames = []
    net_detection_started = False
    ball_center_points_in_rim = []
    ball_center_points_after_rim = [] 
    wait_net_start = False
    
    detected_rim_box = None
    detected_rim_yolo_box = None
    rim_locked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        clean_frame = frame.copy()

        results = model(frame, verbose=False, imgsz=640)
        boxes = results[0].boxes

        rim_candidates = []
        ball_candidates = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == RIM_CLASS_INDEX and conf >= MIN_RIM_CONF:
                rim_candidates.append((box.xyxy[0].cpu().numpy(), box))
            elif cls_id == BASKETBALL_CLASS_INDEX and conf >= MIN_BALL_CONF:
                pt = handle_detect_trajectory_point(frame_count, box)
                if pt is not None:
                    ball_candidates.append((pt, box.xyxy[0].cpu().numpy()))

        if ball_candidates:
            last_point, last_ball_xyxy = ball_candidates[-1]
        
        # Wybór kosza
        if not rim_locked and rim_candidates:
            detected_rim_box, detected_rim_yolo_box = handle_detect_rim(rim_candidates)
            rim_locked = detected_rim_box is not None

        if detected_rim_box is not None:
            tracking_roi = create_tracking_roi(detected_rim_box, frame.shape)

        # --- ANALIZA PIŁKI W OBRĘCZY ---
        ball_in_rim_now = False
        if detected_rim_box is not None and last_ball_xyxy is not None:
            ball_in_rim_now = handle_calculate_ball_in_rim(detected_rim_box, last_ball_xyxy)
        
        if ball_in_rim_now and last_ball_xyxy is not None and last_point is not None:
            ball_center_points_in_rim.append(last_point)

        left_rim_event = ball_in_rim_prev and not ball_in_rim_now
        ball_in_rim_prev = ball_in_rim_now

        # Przejście do stanu WAIT_NET
        if left_rim_event:
            wait_net_start = True
            net_moved_detection_cooldown = NET_MOVED_DETECTION_DURATION
            net_moved_in_frames = []
            net_detection_started = False
            
            if shotState == "TRACKING":
                shotState = "WAIT_NET"

        # Sprawdzenie czy piłka jest w ROI
        in_tracking_roi = False
        if tracking_roi is not None and last_ball_xyxy is not None:
            in_tracking_roi = is_ball_in_roi(last_ball_xyxy, tracking_roi)

        
        if shotState == "IDLE":
            if shot_cooldown <= 0 and in_tracking_roi:
                trajectory_points = []
                ball_center_points_in_rim = []
                ball_center_points_after_rim = []
                shotState = "TRACKING"
                if last_point is not None:
                    trajectory_points.append(last_point)

        elif shotState == "TRACKING":
            if in_tracking_roi:
                if last_point is not None:
                    trajectory_points.append(last_point)
            else:
                # ŚCIEŻKA 1: Piłka wyleciała z ROI (np. pudło, odbicie) - brak fazy WAIT_NET
                # To jest odpowiednik "else" z linii 145 w main.py
                
                features = feature_extraction(trajectory_points, detected_rim_box, fps, ball_center_points_after_rim=[])
                
                if features:
                    features.update({
                        "file_name": video_path.name,
                        "label": "miss", # Domyślnie miss, chyba że folder mówi inaczej, ale w tej ścieżce zazwyczaj to miss
                        "net_moved": False
                    })
                    all_shots_data.append(features)
                    print(f"  -> Zapisano rzut (wyjście z ROI): {video_path.name}")

                # Reset
                trajectory_points = []
                ball_center_points_in_rim = []
                ball_center_points_after_rim = []
                net_moved_in_frames = []
                net_detection_started = False
                wait_net_start = False
                ball_in_rim_prev = False
                shot_net_moved = False
                
                shot_cooldown = COOLDOWN_DURATION
                shotState = "COOLDOWN"

        elif shotState == "WAIT_NET":
            if wait_net_start and last_point is not None:
                ball_center_points_after_rim.append(last_point)

            # Start detekcji gdy piłka spadnie poniżej obręczy
            if wait_net_start and last_ball_xyxy is not None and detected_rim_box is not None:
                bx1, by1, bx2, by2 = last_ball_xyxy
                ball_center_y = (by1 + by2) * 0.5
                rim_bottom_y = detected_rim_box[3]

                if ball_center_y > rim_bottom_y:
                    net_detection_started = True

            # Zbieranie klatek ruchu siatki
            if net_moved_detection_cooldown > 0:
                net_moved_detection_cooldown -= 1
                if net_detection_started and detected_rim_yolo_box is not None and prev_frame is not None:
                    net_attr = handle_detect_net_moved(clean_frame, prev_frame, detected_rim_yolo_box, last_ball_xyxy, ball_center_points_in_rim)
                    net_moved_in_frames.append(net_attr[0] if net_attr else False)
            
            # Koniec czasu detekcji - ŚCIEŻKA 2: Piłka przeszła przez obręcz
            if net_moved_detection_cooldown == 0:
                count_moves = sum(net_moved_in_frames)
                required_moves = 2 if fps <= 35 else 3
                shot_net_moved = count_moves >= required_moves

                features = feature_extraction(trajectory_points, detected_rim_box, fps, ball_center_points_after_rim)
                
                if features:
                    # Automatyczne nadawanie labela na podstawie folderu (opcjonalne)
                    label = "hit" if "trafione" in str(video_path.parent) else "miss"
                    
                    features.update({
                        "file_name": video_path.name,
                        "label": label, 
                        "net_moved": shot_net_moved
                    })
                    all_shots_data.append(features)
                    print(f"  -> Zapisano rzut (pełna analiza): {video_path.name} | Net: {shot_net_moved}")

                # Reset
                trajectory_points = []
                net_moved_in_frames = []
                net_detection_started = False
                wait_net_start = False
                ball_in_rim_prev = False
                ball_center_points_in_rim = []
                
                shot_cooldown = COOLDOWN_DURATION
                shotState = "COOLDOWN"

        elif shotState == "COOLDOWN":
            if shot_cooldown > 0: shot_cooldown -= 1
            else: shotState = "IDLE"

        prev_frame = clean_frame.copy()

    cap.release()

# --- ZAPIS CSV ---
if all_shots_data:
    df = pd.DataFrame(all_shots_data)
    # Uzupełnienie brakujących kolumn, jeśli jakiekolwiek są
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
            
    df = df.reindex(columns=REQUIRED_COLUMNS)
    df.to_csv(csv_output_path, index=False)
    print(f"\nGotowe! Zapisano {len(df)} rekordów do pliku: {csv_output_path}")
else:
    print("\nBrak danych do zapisu. Sprawdź ścieżki lub parametry detekcji.")