from pathlib import Path
import pandas as pd
import joblib
import cv2 as cv
from ultralytics import YOLO
from video_analyzing_scripts.feature_engineering import feature_extraction
from video_analyzing_scripts.rim import handle_detect_net_moved, handle_draw_net_moved, handle_draw_rim, handle_detect_rim, handle_calculate_ball_in_rim
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point, handle_draw_trajectory, handle_draw_ball
from video_analyzing_scripts.throw_roi import create_tracking_roi, is_ball_in_roi
from constants import RIM_CLASS_INDEX, MIN_RIM_CONF, BASKETBALL_CLASS_INDEX, MIN_BALL_CONF
project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v7/weights/best.pt"
lr_model_path = project_dir / "models/lr_model.pkl"
# video_path = project_dir / "data/collection/trafione/scored13.mp4"
# video_path = project_dir / "data/collection/odbite_od_tablicy/backhit27.mp4"
# video_path = project_dir / "data/collection/nietrafione/missed25.mp4"
video_path = project_dir / "data/collection/videos/test/video12.mp4"
output_path = project_dir / "analysis_results" / f"{video_path.stem}_analyzed.mp4"

model = YOLO(model_path)
lr_model = joblib.load(lr_model_path)

basketball_class_index = 0
rim_class_index = 0

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*"mp4v")

frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

out = cv.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

# zmienne do określenia trajektorii ruchu piłki
trajectory_points = []
frame_count = 0
prev_frame = None
point = None

## zmienne do logiki rzutow
shot_cooldown = 0 #licznik przerwy miedzy rzutami
COOLDOWN_DURATION = int(fps * 2) #przerwa miedzy sledzeniem
is_tracking_shot = False #czy sledzenie jest wlaczone
tracking_roi = None # obszar region of intrest
total_shots = 0

# zmienne do detekcji ruchu siatki
net_moved_detection_cooldown = 0
shot_net_moved = False
NET_MOVED_DETECTION_DURATION = int(fps * 0.2) # około 200ms 
basketball_state = "in_air" # "in_air" -> rzut, "in_rim" -> jest w bounding boxie siatki, "out_rim" -> piłka wyszła z bb siatki
net_moved_in_frames = [] # tablica która będzie przechowywała bool'a czy dla danych klatek zaobserwowano ruch siatki
net_detection_started = False
ball_center_points_in_rim = [] # tablica potrzebna do obliczenia najmniejszej odległości od środa obręczy, na tej podstawie będzie obliczany wspólczynnik ruchu siatki

# zmienne do detekcji kosza oraz wybrania prawidłowego
detected_rim_box = None #[xmin, ymin, xmax, ymax]
detected_rim_yolo_box = None # wykryty bb kosza 
rim_locked = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Film nie posiada więcej klatek do wyświetlenia")
        break

    frame_count += 1
    clean_frame = frame.copy()
    
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
        
        out.write(frame)
        cv.imshow("Analiza rzutu", frame)
        if cv.waitKey(1) == ord('q'): 
            break
        continue

    # detekcja
    results = model(frame, verbose=False, imgsz=640)
    boxes = results[0].boxes

    net_attr = None
    current_ball_box = None

    rim_candidates = []
    ball_candidates = []

    # zrobiłem tak, że na początku zbieram wszystkie wykryte kosze a dopiero potem wybieram odpowiedni
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # RIM
        if cls_id == RIM_CLASS_INDEX and conf >= MIN_RIM_CONF:
            rim_xyxy = box.xyxy[0].cpu().numpy()
            rim_candidates.append((rim_xyxy, box))

        # BASKETBALL
        elif cls_id == BASKETBALL_CLASS_INDEX and conf >= MIN_BALL_CONF:
            point = handle_detect_trajectory_point(frame_count, box)
            if point is not None:
                ball_xyxy = box.xyxy[0].cpu().numpy()
                ball_candidates.append((point, ball_xyxy))

    # tutaj wybieramy ten którego pole jest największe, czyli teoretycznie będzie bliżej na filmiku 
    if not rim_locked and rim_candidates:
        detected_rim_box, detected_rim_yolo_box = handle_detect_rim(rim_candidates)
        rim_locked = detected_rim_box is not None

    # jak wykryjemy kosz to rysujemy mu obramowanie, oraz tworzymy tracking roi dla piłki 
    if detected_rim_box is not None:
        handle_draw_rim(frame, detected_rim_box)
        tracking_roi = create_tracking_roi(detected_rim_box, frame.shape)
    
    # detekcja ruchu siatki tylko dla wybranego kosza
    if detected_rim_yolo_box is not None and detected_rim_box is not None:
        b_xyxy = None

        if len(ball_candidates) > 0:
            _, b_xyxy = ball_candidates[-1]

        if b_xyxy is not None:
            in_rim = handle_calculate_ball_in_rim(detected_rim_box, b_xyxy)
            
            if in_rim and basketball_state == "in_air":
                basketball_state = "in_rim"
            elif basketball_state == "in_rim":
                ball_center_points_in_rim.append(b_xyxy)
                print("Dodano")
            elif not in_rim and basketball_state == "in_rim":
                basketball_state = "out_rim"
                net_moved_detection_cooldown = NET_MOVED_DETECTION_DURATION
                net_moved_in_frames = [] 
                net_detection_started = False

        if basketball_state == "out_rim":
            if not net_detection_started and b_xyxy is not None:
                bx1, by1, bx2, by2 = b_xyxy
                ball_center_y = (by1 + by2) * 0.5

                rx1, ry1, rx2, ry2 = detected_rim_box
                rim_bottom_y = ry2

                if ball_center_y > rim_bottom_y:
                    net_detection_started = True

            if net_moved_detection_cooldown > 0 and net_detection_started:
                net_moved_detection_cooldown -= 1
                net_attr = handle_detect_net_moved(clean_frame, prev_frame, detected_rim_yolo_box, b_xyxy, ball_center_points_in_rim)

                if net_attr is not None:
                    is_moving, _ = net_attr
                    net_moved_in_frames.append(bool(is_moving))
                else:
                    net_moved_in_frames.append(False)

            if net_moved_detection_cooldown == 0 and len(net_moved_in_frames) > 0:
                count_of_moved_nets = sum(net_moved_in_frames)
                required_moves = 2 if fps <= 35 else 3
                print(f"Liczba ruchów: {count_of_moved_nets}")
                print(f"Wymagana liczba ruchów: {required_moves}")

                shot_net_moved = count_of_moved_nets >= required_moves
                basketball_state = "in_air"

    # logika piłki, tylko teraz nie obliczam tutaj current_ball_box tylko xyxy piłki jest przekazywane do ball candidates
    for point, current_ball_box in ball_candidates:
        # sprawdzamy czy pilka w ROI
        in_zone = tracking_roi and is_ball_in_roi(current_ball_box, tracking_roi)
        handle_draw_ball(frame, current_ball_box)

        if in_zone:
            if not is_tracking_shot:
                is_tracking_shot = True
                trajectory_points = [] # reset dla nowego rzutu
            trajectory_points.append(point)
        else:
            # Sprawdzenie czy pilka poza strefa -> koniec rzutu
            if is_tracking_shot:
                if basketball_state == "out_rim" and net_moved_detection_cooldown > 0:
                    continue
                
                total_shots += 1
                print(f"Rzut nr {total_shots} zakończony. Analiza...")

                features = feature_extraction(trajectory_points, detected_rim_box, fps)
                features["net_moved"] = shot_net_moved
                features_df = pd.DataFrame([features])
                model_input = features_df[["min_odleglosc_pix", "czy_w_tunelu_pod_obrecza"]]

                prediction = lr_model.predict(model_input)[0]
                if prediction == 'hit':
                    result_text = "TRAFIONY!"
                    color = (0, 255, 0) 
                else:
                    result_text = "PUDŁO"
                    color = (0, 0, 255) 

                print(f"Wynik LR: {result_text}")
                print(f"Dane wejściowe: Dystans={features['min_odleglosc_pix']} | Tunel={features['czy_w_tunelu_pod_obrecza']} | Siatka={features["net_moved"]}")

                # Reset
                is_tracking_shot = False
                shot_net_moved = False
                shot_cooldown = COOLDOWN_DURATION
                trajectory_points = []

                net_moved_detection_cooldown = 0
                basketball_state = "in_air" 
                net_moved_in_frames = []
                net_detection_started = False
        
        # specjalnie ogranicze do jednej piłki narazie
        break

    # rysowanie ROI
    if tracking_roi:
        color = (0, 255, 0) if is_tracking_shot else (255, 0, 0) # Zielony jak nagrywa, niebieski jak czuwa
        cv.rectangle(frame, (tracking_roi[0], tracking_roi[1]), (tracking_roi[2], tracking_roi[3]), color, 2)
        cv.putText(frame, "ROI", (tracking_roi[0], tracking_roi[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # if net_attr is not None:
    #     handle_draw_net_moved(frame, net_attr[0], net_attr[1])
    
    cv.putText(frame, f"State: {basketball_state}", (20, 140), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if basketball_state == "out_rim":
        cv.putText(frame, f"OutRimTimer: {net_moved_detection_cooldown}", (20, 170), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if is_tracking_shot:
        handle_draw_trajectory(frame, trajectory_points)

    out.write(frame)
    cv.imshow("Analiza rzutu", frame)
    prev_frame = clean_frame.copy()

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()