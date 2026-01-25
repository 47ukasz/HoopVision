from pathlib import Path
import pandas as pd
import joblib
import cv2 as cv
from ultralytics import YOLO

from video_analyzing_scripts.feature_engineering import feature_extraction
from video_analyzing_scripts.rim import handle_detect_net_moved, handle_draw_rim, handle_detect_rim, handle_calculate_ball_in_rim
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point, handle_draw_trajectory, handle_draw_ball
from video_analyzing_scripts.throw_roi import create_tracking_roi, is_ball_in_roi
from constants import RIM_CLASS_INDEX, MIN_RIM_CONF, BASKETBALL_CLASS_INDEX, MIN_BALL_CONF

def analyze(
    source,
    output_path: Path | None = None,
    frame_callback = None,
    stats_callback = None,
    stop_flag=None, 
):

    project_dir = Path.cwd()
    print(project_dir)
    model_path = project_dir / "models/hoopvision_v7/weights/best.pt"
    lr_model_path = project_dir / "models/lr_model.pkl"
    #video_path = project_dir / "data/collection/videos/trafione/scored1.mp4"
    # video_path = project_dir / "data/collection/odbite_od_tablicy/backhit27.mp4"
    #video_path = project_dir / "data/collection/videos/long/video12.mp4"
    # video_path = project_dir / "data/collection/videos/test/video9.mp4"

    if isinstance(source, (str, Path)):
        video_path = Path(source)
        cap = cv.VideoCapture(str(video_path))
        if output_path is None:
            output_path = project_dir / "analysis_results" / f"{video_path.stem}_analyzed.mp4"
    else:
        # kamera
        cap = cv.VideoCapture(int(source))
        if output_path is None:
            output_path = project_dir / "analysis_results" / "camera_analyzed.mp4"
    
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć źródła wideo: {source}")

    model = YOLO(model_path)
    lr_model = joblib.load(lr_model_path)

    basketball_class_index = 0
    rim_class_index = 0

    fps = cap.get(cv.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0 #fallback dla kamer

    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    # stan rzutu (to nowe)
    # IDLE, TRACKING, WAIT_NET, COOLDOWN
    shotState = "IDLE" # zmienna do stanu rzutu, giga wazna bo wczesniej cooldown odcinal detekcje czy sie siatka ruszyla

    # zmienne do określenia trajektorii ruchu piłki
    trajectory_points = []
    last_ball_xyxy = None
    last_point = None
    frame_count = 0
    prev_frame = None
    point = None

    ## zmienne do logiki rzutow
    shot_cooldown = 0 #licznik przerwy miedzy rzutami
    COOLDOWN_DURATION = int(fps * 2) #przerwa miedzy sledzeniem
    tracking_roi = None # obszar region of intrest
    total_shots = 0
    hit_shots = 0

    # zmienne do detekcji ruchu siatki
    ball_in_rim_prev = False
    net_moved_detection_cooldown = 0
    shot_net_moved = False
    NET_MOVED_DETECTION_DURATION = int(fps * 0.2) # około 200ms 
    net_moved_in_frames = [] # tablica która będzie przechowywała bool'a czy dla danych klatek zaobserwowano ruch siatki
    net_detection_started = False
    ball_center_points_in_rim = [] # tablica potrzebna do obliczenia najmniejszej odległości od środa obręczy, na tej podstawie będzie obliczany wspólczynnik ruchu siatki
    ball_center_points_after_rim = []
    wait_net_start = False

    # zmienne do detekcji kosza oraz wybrania prawidłowego
    detected_rim_box = None #[xmin, ymin, xmax, ymax]
    detected_rim_yolo_box = None # wykryty bb kosza 
    rim_locked = False

    # zmienne do naprawy gdy piłka wyjdzie poza kadr
    ball_missing_frames = 0

    while cap.isOpened():

        # mozliwosc przerwania z ui
        if stop_flag is not None and getattr(stop_flag, "stop", False):
            break

        ret, frame = cap.read()
        if not ret:
            print("Film nie posiada więcej klatek do wyświetlenia")
            break

        frame_count += 1
        clean_frame = frame.copy()

        # licznik rzutow
        #cv.rectangle(frame, (0, 0), (250, 60), (0, 0, 0), -1)
        #cv.putText(frame, f"Rzuty: {total_shots}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # stan rzutu
        #cv.putText(frame, f"STATE: {shotState}", (20, 95), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        
        # detekcja
        results = model(frame, verbose=False, imgsz=640, device='mps')
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

        # jak yolo nie wykryje pileczki w klatce to zawsze mamy ostatnia, przez co zredukuje sie liczba dziwnych przeskokow
        if ball_candidates:
            last_point, last_ball_xyxy = ball_candidates[-1]

        if last_ball_xyxy is not None:
            handle_draw_ball(frame, last_ball_xyxy)

        # tutaj wybieramy ten którego pole jest największe, czyli teoretycznie będzie bliżej na filmiku 
        if not rim_locked and rim_candidates:
            detected_rim_box, detected_rim_yolo_box = handle_detect_rim(rim_candidates)
            rim_locked = detected_rim_box is not None

        # jak wykryjemy kosz to rysujemy mu obramowanie, oraz tworzymy tracking roi dla piłki 
        if detected_rim_box is not None:
            handle_draw_rim(frame, detected_rim_box)
            tracking_roi = create_tracking_roi(detected_rim_box, frame.shape)
        
        # rysowanie roi
        if tracking_roi:
            color = (0, 255, 0) if shotState == "TRACKING" else (255, 0, 0) # Zielony jak nagrywa, niebieski jak czuwa
            cv.rectangle(frame, (tracking_roi[0], tracking_roi[1]), (tracking_roi[2], tracking_roi[3]), color, 2)
            cv.putText(frame, "ROI", (tracking_roi[0], tracking_roi[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if shotState == "TRACKING" and trajectory_points:
            handle_draw_trajectory(frame, trajectory_points)

        # tutaj zbieramy punkty pilki gdy byla w obreczy, po to zeby obliczyc dystans od srodka i na bazie jego dobrac minialny wspolczynik ruchu siatki
        ball_in_rim_now = False
        if detected_rim_box is not None and last_ball_xyxy is not None:
            ball_in_rim_now = handle_calculate_ball_in_rim(detected_rim_box, last_ball_xyxy)

        if ball_in_rim_now and last_ball_xyxy is not None:
            bx1, by1, bx2, by2 = last_ball_xyxy
            cx = (bx1 + bx2) * 0.5
            cy = (by1 + by2) * 0.5
            ball_center_points_in_rim.append(last_point)

        # zmienne do okreslenia czy pilka byla w siatce i ja opuscila
        left_rim_event = ball_in_rim_prev is True and ball_in_rim_now is False
        ball_in_rim_prev = ball_in_rim_now

        # jezeli pileczka byla w koszu to odpalamy nowy stan dla detekcji siatki i resetujemy zmienne gdyby to nie byl pierwszy rzut
        if left_rim_event:
            wait_net_start = True
            net_moved_detection_cooldown = NET_MOVED_DETECTION_DURATION
            net_moved_in_frames = []
            net_detection_started = False

            if shotState == "TRACKING":
                shotState = "WAIT_NET"

        # teraz tutaj jest core jak cos, obsluga stanow dla rzutu, moze 4 w sumie przyjmowac: IDLE - STAN MIEDZY RZUTAMI, TRACKING - SLEDZI PILKE W ROI, WAIT_NET - OBLICZA CZY SIE SIATKA RUSZYLA I TEZ PO RZUCIE DOKONUJE PREDYCKJI, COOLDOWN - PRZERWA DO KOLEJNEGO RZUTU
        in_tracking_roi = False
        if tracking_roi is not None and last_ball_xyxy is not None:
            in_tracking_roi = is_ball_in_roi(last_ball_xyxy, tracking_roi)
        
        if shotState == "IDLE":
            # to jest stan miedzy cooldownem a nowym rzutem, sluzy do resetowania zmiennych etc.
            if shot_cooldown <= 0 and in_tracking_roi:
                trajectory_points = []
                ball_center_points_in_rim = []
                shotState = "TRACKING" # przejscie do stany sledzenia pilki w roi

                if last_point is not None:
                    trajectory_points.append(last_point)
        elif shotState == "TRACKING":
            # stan gdy pilka jest w roi
            if in_tracking_roi:
                # standardowo, gdy pileczka jest w roi dodajemy punkt do trajektorii
                if last_point is not None:
                    trajectory_points.append(last_point)
            else:
                # pileczka nigdy nie przeleciala przez siatke, wiec rozpoczynamy predykcje 
                total_shots += 1
                print(f"Rzut nr {total_shots} zakończony. Analiza...")

                features = feature_extraction(trajectory_points, detected_rim_box, fps, ball_center_points_after_rim=[])

                if features is None:
                    print("Błąd ekstrakcji danych.")
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
                    # skip tej klatki 
                    prev_frame = clean_frame.copy()
                    continue

                features["net_moved"] = False

                features_df = pd.DataFrame([features])
                model_input = features_df[["min_odleglosc_pix", "kat", "czy_w_tunelu_pod_obrecza", "net_moved"]]
                prediction = lr_model.predict(model_input)[0]

                if prediction == 'hit':
                    result_text = "TRAFIONY!"
                    hit_shots += 1
                else:
                    result_text = "PUDŁO"

                shot_data = {
                    "shot_id": total_shots,
                    "result": prediction,
                    "min_distance_px": features["min_odleglosc_pix"],
                    "angle_deg": features["kat"],
                    "tunnel": features["czy_w_tunelu_pod_obrecza"],
                    "net_moved": shot_net_moved,
                    "net_moves_detected": None,
                    "net_moves_required": None,
                }

                if stats_callback is not None:
                    stats_callback(total_shots, hit_shots, shot_data)


                print(f"Wynik LR: {result_text}")
                print(f"Dane wejściowe: Dystans={features['min_odleglosc_pix']} | Tunel={features['czy_w_tunelu_pod_obrecza']} | Siatka={features['net_moved']}")

                # RESET po rzucie
                trajectory_points = []
                ball_center_points_in_rim = []
                ball_center_points_after_rim = []
                net_moved_in_frames = []
                net_detection_started = False
                wait_net_start = False
                ball_in_rim_prev = False
                shot_net_moved = False

                # cooldown
                shot_cooldown = COOLDOWN_DURATION
                shotState = "COOLDOWN"
        elif shotState == "WAIT_NET":
            if wait_net_start and last_point is not None:
                ball_center_points_after_rim.append(last_point)
                
            # i to jest teraz stan do detekcji siatki, bugfixuje to ze wczesniej gdy pilka wypadala z roi to od razu byl cooldown i nie bylo czasu zeby sprawdzic czy sie siatka ruszyla
            if wait_net_start and last_ball_xyxy is not None and detected_rim_box is not None:
                # obliczamy czy pilka jest ponizej kosza, jezeli tak zaczynamy detekcje
                bx1, by1, bx2, by2 = last_ball_xyxy
                ball_center_y = (by1 + by2) * 0.5

                rx1, ry1, rx2, ry2 = detected_rim_box
                rim_bottom_y = ry2

                if ball_center_y > rim_bottom_y:
                    net_detection_started = True
            
            # czas w ktorym zbieramy dane do okreslenia czy sie siatka ruszyla
            if net_moved_detection_cooldown > 0:
                net_moved_detection_cooldown -= 1

                if net_detection_started and detected_rim_yolo_box is not None and prev_frame is not None:
                    net_attr = handle_detect_net_moved(clean_frame, prev_frame, detected_rim_yolo_box, last_ball_xyxy, ball_center_points_in_rim)

                    if net_attr is not None:
                        is_moving, _ = net_attr
                        net_moved_in_frames.append(is_moving)
                    else:
                        net_moved_in_frames.append(False)
            
            # koniec zbierania danych, wiec sprawdzamy czy siatka ruszyla sie wystarczajaca ilosc razy
            if net_moved_detection_cooldown == 0:
                count_of_moved_nets = sum(net_moved_in_frames)
                required_moves = 2 if fps <= 35 else 3
                print(f"Liczba ruchów: {count_of_moved_nets}")
                print(f"Wymagana liczba ruchów: {required_moves}")

                shot_net_moved = count_of_moved_nets >= required_moves

                # to jest juz etap po rzucie wiec teraz jest predykcja

                total_shots += 1
                print(f"Rzut nr {total_shots} zakończony (bez siatki i tunelu). Analiza...")
                
                features = feature_extraction(trajectory_points, detected_rim_box, fps, ball_center_points_after_rim)

                if features is None:
                    print("Błąd ekstrakcji danych.")
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
                    # skip tej klatki 
                    prev_frame = clean_frame.copy()
                    continue

                features["net_moved"] = shot_net_moved
                
                features_df = pd.DataFrame([features])
                model_input = features_df[["min_odleglosc_pix", "kat", "czy_w_tunelu_pod_obrecza", "net_moved"]]
                prediction = lr_model.predict(model_input)[0]

                if prediction == 'hit':
                    result_text = "TRAFIONY!"
                    hit_shots += 1
                else:
                    result_text = "PUDŁO"

                shot_data = {
                    "shot_id": total_shots,
                    "result": prediction,
                    "min_distance_px": features["min_odleglosc_pix"],
                    "angle_deg": features["kat"],
                    "tunnel": features["czy_w_tunelu_pod_obrecza"],
                    "net_moved": shot_net_moved,
                    "net_moves_detected": None,
                    "net_moves_required": None,
                }

                if stats_callback is not None:
                    stats_callback(total_shots, hit_shots, shot_data)


                print(f"Wynik LR: {result_text}")
                print(f"Dane wejściowe: Dystans={features['min_odleglosc_pix']} | Tunel={features['czy_w_tunelu_pod_obrecza']} | Siatka={features['net_moved']}")

                # RESET

                trajectory_points = []
                net_moved_detection_cooldown = 0
                net_moved_in_frames = []
                net_detection_started = False
                wait_net_start = False
                ball_in_rim_prev = False
                ball_center_points_in_rim = []

                # ustawienie cooldownu po rzucie

                shot_cooldown = COOLDOWN_DURATION
                shotState = "COOLDOWN"
        elif shotState == "COOLDOWN":
            if shot_cooldown > 0:
                shot_cooldown -= 1
                cv.putText(frame, f"Cooldown: {shot_cooldown}", (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)        
            else:
                shotState = "IDLE" # przejscie do stanu miedzy rzutami

        out.write(frame)
        if frame_callback is not None:
            frame_callback(frame)
        else:
            cv.imshow("Analiza rzutu", frame)
            if cv.waitKey(1) == ord('q'):
                break
        
        prev_frame = clean_frame.copy()

    cap.release()
    out.release()
    cv.destroyAllWindows()