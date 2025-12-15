# -*- coding: utf-8 -*-

from pathlib import Path
import csv

import cv2 as cv
from ultralytics import YOLO

from constants import RIM_CLASS_INDEX, MIN_RIM_CONF
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point
from video_analyzing_scripts.feature_engineering import feature_extraction


# -------------------
# KONFIGURACJA
# -------------------

# Katalog główny projektu (HoopVision), niezależnie od tego skąd odpalasz
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Model YOLO (aktualny)
MODEL_PATH = PROJECT_DIR / "models" / "hoopvision_v7" / "weights" / "best.pt"

# Katalogi z filmami + etykiety
VIDEO_DIRS = {
    "hit": PROJECT_DIR / "data" / "collection" / "trafione",
    "miss": PROJECT_DIR / "data" / "collection" / "nietrafione",
    "off_backboard": PROJECT_DIR / "data" / "collection" / "odbite od tablicy",
}

# Output CSV
OUTPUT_CSV = PROJECT_DIR / "data" / "trajectory_features.csv"

# Minimalna liczba punktów trajektorii, żeby liczyć cechy
MIN_TRAJECTORY_POINTS = 6

# Próg do detekcji ruchu siatki (średnia różnica pikseli w ROI obręczy)
NET_MOVEMENT_THRESHOLD = 15

# Jeśli True i plik CSV już istnieje: pomijaj filmy już zapisane w CSV
RESUME_IF_CSV_EXISTS = True

# Jeśli True: wycisza spam logów YOLO (recommended)
YOLO_VERBOSE = False


# -------------------
# FUNKCJE POMOCNICZE
# -------------------

def load_model(model_path: Path) -> YOLO:
    print(f"[INFO] Ładowanie modelu z: {model_path}")
    return YOLO(str(model_path))


def detect_net_movement(previous_frame, current_frame, rim_box, threshold=NET_MOVEMENT_THRESHOLD) -> bool:
    """
    Wykrywa ruch siatki na podstawie różnicy pikselowej w obszarze obręczy (rim_box).
    Działa OFFLINE: nie wymaga YOLO boxów (brak .cls), tylko [x1,y1,x2,y2].

    Mechanizm:
      - wycinamy ROI obręczy z dwóch kolejnych klatek,
      - absdiff,
      - średnia z grayscale,
      - jeśli przekracza próg -> net_moved = True.
    """
    if previous_frame is None or current_frame is None or rim_box is None:
        return False

    x1, y1, x2, y2 = map(int, rim_box)

    cut_prev = previous_frame[y1:y2, x1:x2]
    cut_curr = current_frame[y1:y2, x1:x2]

    if cut_prev.size == 0 or cut_curr.size == 0:
        return False

    diff = cv.absdiff(cut_prev, cut_curr)
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    return diff_gray.mean() > threshold


def analyze_single_video(video_path: Path, label: str, model: YOLO):
    """
    Przetwarza jeden film:
      - wyciąga trajektorię piłki (punkty),
      - wykrywa obręcz (rim_box),
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
    if fps is None or fps <= 0:
        print(f"[WARN] Nieprawidłowy FPS dla: {video_path.name}")
        cap.release()
        return None

    frame_id = 0
    trajectory_points = []   # [(frame_id, x, y), ...]
    rim_box = None           # [x1, y1, x2, y2]
    net_moved = False

    previous_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # YOLO predykcja na klatce
        results = model(frame, verbose=YOLO_VERBOSE)[0]
        boxes = results.boxes

        # --- DETEKCJA RUCHU SIATKI ---
        # W batchu nie używamy "pełnych YOLO boxów siatki", tylko ROI obręczy (rim_box),
        # bo chcemy wyłącznie flagę: czy siatka ruszyła się w trakcie rzutu.
        if rim_box is not None and previous_frame is not None and not net_moved:
            moved = detect_net_movement(previous_frame, frame, rim_box)
            if moved:
                net_moved = True

        previous_frame = frame  # nie robimy copy() - wystarczy referencja do aktualnej macierzy

        # --- ANALIZA BOXÓW ---
        for box in boxes:
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])

            # piłka (delegujemy do istniejącej funkcji)
            traj_point = handle_detect_trajectory_point(frame_id, box)
            if traj_point is not None:
                trajectory_points.append(traj_point)

            # obręcz – aktualizujemy rim_box na bieżąco (bierzemy ostatnią stabilną detekcję)
            if cls_idx == RIM_CLASS_INDEX and conf >= MIN_RIM_CONF:
                xyxy = box.xyxy[0].cpu().numpy()
                rim_box = xyxy.tolist()

    cap.release()

    # --- WALIDACJA ---
    if rim_box is None or len(trajectory_points) < MIN_TRAJECTORY_POINTS:
        print(f"[WARN] Za mało danych (obręcz/trajectoria) dla: {video_path.name}")
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
        "net_moved": net_moved,
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


def load_already_processed(csv_path: Path) -> set[str]:
    """
    Jeśli CSV istnieje i RESUME_IF_CSV_EXISTS=True,
    zwraca zbiór nazw plików, które już są w CSV (kolumna file_name).
    """
    processed = set()
    if not csv_path.exists():
        return processed

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "file_name" not in (reader.fieldnames or []):
                return processed
            for row in reader:
                if row.get("file_name"):
                    processed.add(row["file_name"])
    except Exception as e:
        print(f"[WARN] Nie mogę odczytać istniejącego CSV ({csv_path}): {e}")

    return processed


def save_csv(rows, csv_path: Path):
    """
    Zapisuje CSV od zera (nadpisuje plik).
    """
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

    processed_files = set()
    if RESUME_IF_CSV_EXISTS:
        processed_files = load_already_processed(OUTPUT_CSV)
        if processed_files:
            print(f"[INFO] W trybie resume: pomijam {len(processed_files)} plików już w CSV.")

    all_results = []

    for video_path, label in video_files:
        if RESUME_IF_CSV_EXISTS and video_path.name in processed_files:
            print(f"[INFO] Pomijam (już w CSV): {video_path.name}")
            continue

        res = analyze_single_video(video_path, label, model)
        if res is not None:
            all_results.append(res)

    save_csv(all_results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
