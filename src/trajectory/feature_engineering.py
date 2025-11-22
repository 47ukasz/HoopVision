import numpy as np

def feature_extraction(trajectory, rim_box, fps):
    """
    Oblicza statystyki rzutu na podstawie zebranych danych.
    trajectory: lista [(frame_idx, x, y), ...]
    rim_box: [x_min, y_min, x_max, y_max]
    """
    if not trajectory:
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
    
    #predkosc i kąt 
    check_idx = max(1, min_dist_idx - 5)
    vec = points[check_idx] - points[check_idx-1]
    velocity = np.linalg.norm(vec) # pix/klatkę
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    
    #czy konczy pod obrecza
    ends_below = points[-1][1] > r_ymax
    
    return {
        "min_odleglosc_pix": round(min_dist, 2),
        "predkosc": round(velocity, 2),
        "kat": round(angle, 2),
        "czy_pod_obrecza": ends_below,
    }
