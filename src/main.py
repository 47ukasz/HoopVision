from pathlib import Path
import cv2 as cv
from ultralytics import YOLO
import time
import csv

from video_analyzing_scripts.feature_engineering import handle_display_features
from video_analyzing_scripts.rim import handle_detect_net_moved, handle_detetect_rim, handle_draw_net_moved
from video_analyzing_scripts.trajectory import handle_detect_trajectory_point, handle_draw_trajectory

project_dir = Path.cwd()
model_path = project_dir / "models/hoopvision_v6/weights/best.pt"
collection_path = project_dir / "data/collection"

csv_output = project_dir / "analysis_results.csv"

SHOW_PREVIEW = False  # <--- ustaw True, jeśli chcesz widok podglądu

model = YOLO(model_path)

def analyze_video(video_path: Path):
    cap = cv.VideoCapture(str(video_path))
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_length_seconds = total_frames / fps if fps else 0.0

    trajectory_points = []
    detected_rim_box = None
    frame_count = 0
    prev_frame = None

    total_inference_time = 0.0
    total_detect_rim_time = 0.0
    total_data_collect_time = 0.0
    total_draw_trajectory_time = 0.0
    total_display_time = 0.0

    global_start = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        inf_start = time.perf_counter()
        results = model(frame)
        inf_end = time.perf_counter()
        total_inference_time += inf_end - inf_start

        boxes = results[0].boxes

        detect_rim_start = time.perf_counter()
        for box in boxes:
            detected_rim_box = handle_detetect_rim(frame, box)
        detect_rim_end = time.perf_counter()
        total_detect_rim_time += (detect_rim_end - detect_rim_start)

        data_collect_start = time.perf_counter()
        for box in boxes:
            point = handle_detect_trajectory_point(frame_count, box)
            if point is not None:
                trajectory_points.append(point)

            if prev_frame is not None:
                net_attr = handle_detect_net_moved(frame, prev_frame, box)
                if net_attr is not None:
                    handle_draw_net_moved(frame, net_attr[0], net_attr[1])
        data_collect_end = time.perf_counter()
        total_data_collect_time += (data_collect_end - data_collect_start)

        draw_traj_start = time.perf_counter()
        handle_draw_trajectory(frame, trajectory_points)
        draw_traj_end = time.perf_counter()
        total_draw_trajectory_time += (draw_traj_end - draw_traj_start)

        if SHOW_PREVIEW:
            display_start = time.perf_counter()
            cv.imshow(f"Analiza: {video_path.name}", frame)
            # waitKey(1) minimalnie blokuje — mierzymy jego czas
            if cv.waitKey(1) == ord('q'):
                display_end = time.perf_counter()
                total_display_time += (display_end - display_start)
                break
            display_end = time.perf_counter()
            total_display_time += (display_end - display_start)

        prev_frame = frame.copy()

    cap.release()
    cv.destroyAllWindows()

    global_end = time.perf_counter()
    total_video_time = global_end - global_start

    def avg(total):
        return (total / frame_count) if frame_count else 0.0

    sum_measured = (
        total_inference_time
        + total_detect_rim_time
        + total_data_collect_time
        + total_draw_trajectory_time
        + total_display_time
    )

    residual_time = total_video_time - sum_measured
    if residual_time < 0 and residual_time > -1e-6:
        residual_time = 0.0

    handle_display_features(trajectory_points, detected_rim_box, fps)

    return {
        "frames": frame_count,
        "video_length": video_length_seconds,
        "total_video_time": total_video_time,
        "avg_time_per_frame": avg(total_video_time),

        "total_inference_time": total_inference_time,
        "avg_inference_per_frame": avg(total_inference_time),

        "total_detect_rim_time": total_detect_rim_time,
        "avg_detect_rim_per_frame": avg(total_detect_rim_time),

        "total_data_collect_time": total_data_collect_time,
        "avg_data_collect_per_frame": avg(total_data_collect_time),

        "total_draw_trajectory_time": total_draw_trajectory_time,
        "avg_draw_trajectory_per_frame": avg(total_draw_trajectory_time),

        "total_display_time": total_display_time,
        "avg_display_per_frame": avg(total_display_time),

        "sum_measured": sum_measured,
        "residual_time": residual_time,
        "avg_residual_per_frame": avg(residual_time),
    }

video_files = list(collection_path.rglob("*.mp4"))

if not video_files:
    print("Brak filmów w folderze collection.")
    exit()

write_header = not csv_output.exists()

with open(csv_output, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    if write_header:
        writer.writerow([
            "video_file",
            "video_length_seconds",
            "frame_count",
            "total_video_time",
            "avg_time_per_frame",
            "total_inference_time",
            "avg_inference_per_frame",
            "total_detect_rim_time",
            "avg_detect_rim_per_frame",
            "total_data_collect_time",
            "avg_data_collect_per_frame",
            "total_draw_trajectory_time",
            "avg_draw_trajectory_per_frame",
            "total_display_time",
            "avg_display_per_frame",
            "sum_measured_time",
            "residual_time",
            "avg_residual_per_frame"
        ])

    for video_path in video_files:
        print(f"\nAnalizuję: {video_path.name}")

        stats = analyze_video(video_path)

        writer.writerow([
            video_path.name,
            stats["video_length"],
            stats["frames"],
            stats["total_video_time"],
            stats["avg_time_per_frame"],
            stats["total_inference_time"],
            stats["avg_inference_per_frame"],
            stats["total_detect_rim_time"],
            stats["avg_detect_rim_per_frame"],
            stats["total_data_collect_time"],
            stats["avg_data_collect_per_frame"],
            stats["total_draw_trajectory_time"],
            stats["avg_draw_trajectory_per_frame"],
            stats["total_display_time"],
            stats["avg_display_per_frame"],
            stats["sum_measured"],
            stats["residual_time"],
            stats["avg_residual_per_frame"]
        ])

        print(f" → Zakończono {video_path.name}")
        print(f"   Całkowity czas: {stats['total_video_time']:.3f}s | Suma mierzonych: {stats['sum_measured']:.3f}s | Residual: {stats['residual_time']:.6f}s")

print("\nAnaliza zakończona. Wyniki zapisano do analysis_results.csv")
