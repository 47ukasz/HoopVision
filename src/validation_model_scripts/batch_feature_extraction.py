from pathlib import Path
import csv

import cv2 as cv
from ultralytics import YOLO

from constants import BASKETBALL_CLASS_INDEX, RIM_CLASS_INDEX, MIN_BALL_CONF, MIN_RIM_CONF
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point
from video_analyzing_scripts.feature_engineering import feature_extraction
from video_analyzing_scripts.rim import handle_detect_net_moved


# -------------------
# KONFIGURACJA
# -------------------

# katalog główny projektu (HoopVision), niezależnie od tego skąd odpalasz
PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_DIR / "models" / "hoopvision_v6" / "weights" / "best.pt"

# katalogi z filmami i ich etykietami
# tutaj możesz sobie dopasować nazwy katalogów
VIDEO_DIRS = {
    "hit": PROJECT_DIR / "data/collection/trafione",
    "miss": PROJECT_DIR / "data/collection/nietrafione",
}

# dokąd zapisywać cechy
OUTPUT_CSV = PROJECT_DIR / "data/trajectory_features.csv"


# -------------------
# FUNKCJE POMOCNICZE
# -------------------

def load_model(model_path: Path) -> YOLO:
    print(f"[INFO] Ładowanie modelu z: {model_path}")
    return YOLO(str(model_path))


def detect_net_movement(previous_frame, current_frame, rim_box, threshold=15):
    """
    Wykrywa ruch siatki na podstawie dwóch klatek i wycinka obręczy.
    NIE używa YOLO box.cls – działa tylko na rim_box (x1,y1,x2,y2).
    """
    if previous_frame is None or current_frame is None:
        return False

    x1, y1, x2, y2 = map(int, rim_box)

    cut_prev = previous_frame[y1:y2, x1:x2]
    cut_curr = current_frame[y1:y2, x1:x2]

    if cut_prev.size == 0 or cut_curr.size == 0:
        return False

    diff = cv.absdiff(cut_prev, cut_curr)
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    mean_diff = diff_gray.mean()

    return mean_diff > threshold


def analyze_single_video(video_path: Path, label: str, model: YOLO):
    """
    Przetwarza jeden film:
    - wyciąga trajektorię,
    - wykrywa obręcz,
    - wykrywa ruch siatki (net_moved),
    - wyciąga cechy,
    - zwraca rekord do CSV.
    """
    print(f"[INFO] Przetwarzam: {video_path.name} (label={label})")

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Nie mogę otworzyć wideo: {video_path}")
        return None

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_id = 0

    trajectory_points = []
    rim_box = None
    net_moved = False

    previous_frame = None  # dla logiki ruchu siatki

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # YOLO predykcja
        results = model(frame)[0]
        boxes = results.boxes

        # --- DETEKCJA RUCHU SIATKI ---
        # W przetwarzaniu masowym nie korzystamy z pełnych YOLO boxów klasy "rim",
        # tylko z wcześniej zapamiętanego rim_box (x1,y1,x2,y2).
        # Wystarczy porównać fragmenty dwóch kolejnych klatek (poprzedniej i obecnej)
        # aby ustalić, czy siatka poruszyła się w czasie rzutu.
        if rim_box is not None and previous_frame is not None:
            # rim_box musi być przekonwertowany do intów, aby użyć go jako zakresów pixeli.
            cords_int = list(map(int, rim_box))

            moved = detect_net_movement(previous_frame, frame, cords_int)
            if moved:
                net_moved = True

        # zapamiętaj aktualną klatkę jako poprzednią
        previous_frame = frame.copy()

        # --- ANALIZA BOXÓW ---
        for box in boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            # piłka
            traj_point = handle_detect_trajectory_point(frame_id, box)
            if traj_point is not None:
                trajectory_points.append(traj_point)

            # obręcz – aktualizujemy rim_box na bieżąco
            if cls_idx == RIM_CLASS_INDEX and conf >= MIN_RIM_CONF:
                rim_box = xyxy.tolist()

    cap.release()

    # --- WALIDACJA ---
    if rim_box is None or len(trajectory_points) <= 5:
        print(f"[WARN] Za mało danych (obręcz/trajectoria) dla: {video_path.name}")
        return None

    if fps is None or fps <= 0:
        print(f"[WARN] Nieprawidłowy FPS dla: {video_path.name}")
        return None

    # --- CECHY ---
    features = feature_extraction(trajectory_points, rim_box, fps)
    if features is None:
        print(f"[WARN] feature_extraction zwróciło None dla: {video_path.name}")
        return None

    # --- REKORD DO CSV ---
    return {
        "file_name": video_path.name,
        "label": label,
        "min_odleglosc_pix": features.get("min_odleglosc_pix"),
        "predkosc": features.get("predkosc"),
        "kat": features.get("kat"),
        "czy_w_tunelu_pod_obrecza": features.get("czy_w_tunelu_pod_obrecza"),
        "net_moved": net_moved  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< NEW
    }



def collect_videos(video_dirs: dict[str, Path]):
    """
    Zwraca listę (ścieżka, etykieta) dla wszystkich plików wideo.
    """
    videos = []
    for label, directory in video_dirs.items():
        if not directory.exists():
            print(f"[WARN] Katalog nie istnieje: {directory}")
            continue

        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for file in directory.glob(ext):
                videos.append((file, label))

    return videos


def save_csv(rows, csv_path: Path):
    if not rows:
        print("[WARN] Brak danych do zapisania.")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Zapisano wyniki do: {csv_path}")


# -------------------
# GŁÓWNA FUNKCJA
# -------------------

def main():
    model = load_model(MODEL_PATH)
    video_files = collect_videos(VIDEO_DIRS)

    print(f"[INFO] Znaleziono {len(video_files)} plików wideo.")

    all_results = []
    for video_path, label in video_files:
        res = analyze_single_video(video_path, label, model)
        if res is not None:
            all_results.append(res)

    save_csv(all_results, OUTPUT_CSV)

if __name__ == "__main__":
    main()
