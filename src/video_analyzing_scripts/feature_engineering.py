import numpy as np

def feature_extraction(trajectory, rim_box, fps, ball_center_points_after_rim):
    """
    Oblicza statystyki rzutu na podstawie zebranych danych.
    trajectory: lista [(frame_idx, x, y), ...]
    rim_box: [x_min, y_min, x_max, y_max]
    """
    if not trajectory or len(trajectory) <= 5:
        return None
    
    if rim_box is None: 
        return None
    
    #konwersja do numpy
    points = np.array([t[1:] for t in trajectory]) #tylko(x,y)
    
    #srodek obreczy
    r_xmin, r_ymin, r_xmax, r_ymax = rim_box
    rim_center = np.array([(r_xmin + r_xmax) / 2, (r_ymin + r_ymax) / 2])
    
    #najmnijesza odleglosc pilki od srodka obreczy 
    dists = np.linalg.norm(points - rim_center, axis = 1)
    min_dist_idx = np.argmin(dists)
    min_dist = dists[min_dist_idx] 

    after_x = None
    after_y = None

    if ball_center_points_after_rim and len(ball_center_points_after_rim) > 0:
        offset_frames = max(0, int(.1 * fps))
        idx = min(len(ball_center_points_after_rim) - 1, offset_frames)

        after_x = float(ball_center_points_after_rim[idx][1])
        after_y = float(ball_center_points_after_rim[idx][2])
    else:
        after_x = float(points[-1][0])
        after_y = float(points[-1][1])
    
    #warunek wysokosci(musi byc pod obrecza)
    is_below_rim = after_y > r_ymax
    
    #warunek szerokosci
    rim_width = r_xmax - r_xmin
    margin = rim_width * 0.15
    is_within_with = (r_xmin - margin) < after_x < (r_xmax + margin)
    
    #warunek czy pilka jest w "tunelu" obreczy po najblizszym spotakniu z centrum obreczy
    is_in_tunel = is_below_rim and is_within_with
    
    #TODO calculate angle and velocity after time_offset
    #predkosc i kąt 
    check_idx = max(1, min_dist_idx - 5)
    vec = points[check_idx] - points[check_idx-1]
    velocity = np.linalg.norm(vec) # pix/klatkę
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    
    return {
        "min_odleglosc_pix": round(min_dist, 2),
        "predkosc": round(velocity, 2),
        "kat": round(angle, 2),
        "czy_w_tunelu_pod_obrecza": is_in_tunel,
    }

def handle_display_features(trajectory_points, detected_rim_box, fps):
    features = feature_extraction(trajectory_points, detected_rim_box, fps)

    if features is not None:
        print(f"Minimalna odległość od centrum: {features['min_odleglosc_pix']} px")
        print(f"Prędkość: {features['predkosc']} px/frame")
        print(f"Kąt: {features['kat']} stopni")
        print(f"Czy skończyło się pod obręczą?: {'TAK' if features['czy_w_tunelu_pod_obrecza'] else 'NIE'}")
    else:
        print("Nie udało się zebrać wystarczających danych (brak obręczy lub krótka trajektoria).")
