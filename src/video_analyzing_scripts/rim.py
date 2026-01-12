import cv2 as cv
import numpy as np
from constants import RIM_CLASS_INDEX, MIN_RIM_CONF

def handle_detect_rim(rim_candidates):
    if not rim_candidates:
        return None
    
    rim_xyxy, rim_box = max(rim_candidates, key=lambda t: handle_calculate_rim_area(t[0]))
    return rim_xyxy, rim_box

def handle_detect_net_moved(currentFrame, prevFrame, rim_box, b_xyxy = None, ball_points = None):
    box_class_index = int(rim_box.cls[0])
    box_conf_level = float(rim_box.conf[0])

    padding = 10
    DEFAULT_MIN_RIM_MOTION = 1.0

    if box_class_index != RIM_CLASS_INDEX:
        return (False, None)
    
    if box_conf_level < MIN_RIM_CONF:
        return (False, None)

    if currentFrame is None or prevFrame is None:
        return (False, None)

    curr_gray = cv.cvtColor(currentFrame, cv.COLOR_BGR2GRAY)
    prev_gray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)

    x1, y1, x2, y2 = map(int, rim_box.xyxy[0])
    h, w = curr_gray.shape[:2]
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    cords = (x1, y1, x2, y2)

    curr_area = curr_gray[y1:y2, x1:x2]
    prev_area = prev_gray[y1:y2, x1:x2]

    if prev_area.size <= 0 or curr_area.size <= 0:
        return (False, None)
    
    flow = cv.calcOpticalFlowFarneback(prev_area, curr_area, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 7, poly_sigma = 1.1, flags = 0)
    mag, _ = cv.cartToPolar(flow[:,:,0], flow[:,:,1])

    # full_mask = handle_make_mask(currentFrame, rim_box, b_xyxy)
    # mask_roi = full_mask[y1:y2, x1:x2]
    # valid = (mask_roi == 255)

    # if not np.any(valid):
        # rim_motion_value = 0.0
    # else:
    rim_motion_value = np.mean(mag)

    distance = handle_calculate_distance_from_rim_center(map(int, rim_box.xyxy[0]), ball_points)

    print(f"Obliczony dystans: {distance}")

    if distance is not None:
        rim_moving = rim_motion_value > DEFAULT_MIN_RIM_MOTION
    else:
        rim_moving = rim_motion_value > DEFAULT_MIN_RIM_MOTION

    return (rim_moving, cords)

def handle_make_mask(frame, net_box, b_xyxy = None, expand=0.6):
    height, width = frame.shape[0:2]
    
    xyxy = net_box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, xyxy)

    mask = np.zeros((height, width), dtype=np.uint8)

    cv.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    if b_xyxy is not None:
        # dla bounding box'a piłki obliczamy jego środek oraz powiększamy go o expand %
        # zabezpieczamy się w ten sposób, że piksele poza bounding box'em piłki jeszcze zawierają ślad przemieszczania piłki
        bx1, by1, bx2, by2 = map(int, b_xyxy)

        b_width = bx2 - bx1
        b_height = by2 - by1

        b_center_x = (bx1 + bx2) * 0.5
        b_center_y = (by1 + by2) * 0.5

        b_width_expanded = b_width * (1.0 + expand)
        b_height_expanded = b_height * (1.0 + expand)

        bx1_expanded = int(b_center_x - b_width_expanded * 0.5)
        by1_expanded = int(b_center_y - b_height_expanded * 0.5)
        bx2_expanded = int(b_center_x + b_width_expanded * 0.5)
        by2_expanded = int(b_center_y + b_height_expanded * 0.5)

        # zabezpiecznie, żeby nie wyrzuciło ujemnych współrzędnych albo spoza zakresu (width, height)
        bx1_expanded = max(0, bx1_expanded)
        by1_expanded = max(0, by1_expanded)
        bx2_expanded = min(width - 1, bx2_expanded)
        by2_expanded = min(height - 1, by2_expanded)

        cv.rectangle(mask, (bx1_expanded, by1_expanded), (bx2_expanded, by2_expanded), 0, -1)

    return mask

def handle_draw_net_moved(currentFrame, rim_moved, cords):
    x1, y1, x2, y2 = cords

    label = f'{"Nie" if not rim_moved else ""} porusza sie'
    label_color = (255, 0, 0) if not rim_moved else (0, 255, 0)
    cv.putText(currentFrame, label, (x1, y1 - 20), cv.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
    
    rim_color = (255, 0, 0) if not rim_moved else (0, 255, 0)

    cv.rectangle(currentFrame, (x1, y1), (x2, y2), rim_color, 2)

def handle_draw_rim(currentFrame, rim_xyxy):
    
    if rim_xyxy is None:
        return None

    x1, y1, x2, y2 = map(int, rim_xyxy)

    # box kosza
    cv.rectangle(currentFrame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # środek kosza
    target_x = int((x1 + x2) / 2)
    target_y = int((y1 + y2) / 2)
    cv.circle(currentFrame, (target_x, target_y), 3, (0, 255, 255), -1)

    # "tunel" / krawędzie
    rim_width = x2 - x1
    margin = int(rim_width * 0.15)

    cv.line(currentFrame, (x1 - margin, y2 - margin), (x1, y2 + 200), (0, 255, 0), 2)
    cv.line(currentFrame, (x2 + margin, y2 + margin), (x2, y2 + 200), (0, 255, 0), 2)

def handle_calculate_rim_area(r_xyxy):
    x1, y1, x2, y2 = map(float, r_xyxy)

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def handle_calculate_ball_in_rim(r_xyxy, b_xyxy):
    if r_xyxy is None or b_xyxy is None:
        return False
    
    rx1, ry1, rx2, ry2 = r_xyxy
    bx1, by1, bx2, by2 = b_xyxy

    b_center_x = (bx1 + bx2) * 0.5
    b_center_y = (by1 + by2) * 0.5

    if rx1 <= b_center_x <= rx2 and ry1 <= b_center_y <= ry2:
        return True
    
    return False        

def handle_calculate_distance_from_rim_center(r_xyxy, ball_points):
    if r_xyxy is None or ball_points is None:
        return None
    
    r_xmin, r_ymin, r_xmax, r_ymax = r_xyxy
    rim_center = np.array([(r_xmin + r_xmax) / 2, (r_ymin + r_ymax) / 2])
    points = np.array([t[1:] for t in ball_points]) #tylko(x,y)

    dists = np.linalg.norm(points - rim_center, axis = 1)
    min_dist_idx = np.argmin(dists)
    min_dist = dists[min_dist_idx]

    return min_dist
