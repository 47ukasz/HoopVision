import cv2 as cv
import numpy as np
from constants import RIM_CLASS_INDEX, MIN_RIM_CONF

def handle_detetect_rim(currentFrame, box):
    box_class_index = int(box.cls[0])
    box_conf_level = float(box.conf[0])
    
    if box_class_index != RIM_CLASS_INDEX:
        return None
    
    if box_conf_level < MIN_RIM_CONF:
        return None
    
    xyxy = box.xyxy[0].cpu().numpy()

    rx1, ry1, rx2, ry2 = map(int, xyxy)
    
    cv.rectangle(currentFrame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        
    target_x = int((rx1 + rx2) / 2)
    target_y = int((ry1 + ry2) / 2) 
    
    cv.circle(currentFrame, (target_x, target_y), 3, (0, 255, 255), -1)   
    
    #krawedzie tunelu
    rim_width = rx2 - rx1
    margin = int(rim_width * 0.15) 
    
    cv.line(currentFrame, (rx1 - margin, ry2 - margin), (rx1, ry2 + 200), (0, 255, 0), 2)
    cv.line(currentFrame, (rx2 + margin, ry2 + margin), (rx2, ry2 + 200), (0, 255, 0), 2)

    return xyxy

def handle_detect_net_moved(currentFrame, prevFrame, box):
    box_class_index = int(box.cls[0])
    box_conf_level = float(box.conf[0])

    margin = 10
    min_rim_motion = 1.0

    if box_class_index != RIM_CLASS_INDEX:
        return
    
    if box_conf_level < MIN_RIM_CONF:
        return

    if currentFrame is None or prevFrame is None:
        return

    curr_gray = cv.cvtColor(currentFrame, cv.COLOR_BGR2GRAY)
    prev_gray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)

    cords = [int(cord) + margin for cord in box.xyxy[0]]
    x1, y1, x2, y2 = cords

    curr_area = curr_gray[y1:y2, x1:x2]
    prev_area = prev_gray[y1:y2, x1:x2]

    if prev_area.size <= 0 and curr_area.size <= 0:
        return
    
    flow = cv.calcOpticalFlowFarneback(prev_area, curr_area, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 7, poly_sigma = 1.1, flags = 0)
    mag, _ = cv.cartToPolar(flow[:,:,0], flow[:,:,1])

    rim_motion_value = np.mean(mag)
    rim_moving = rim_motion_value > min_rim_motion

    return (rim_moving, cords)


def handle_draw_net_moved(currentFrame, rim_moved, cords):
    x1, y1, x2, y2 = cords

    label = f"{"Nie" if not rim_moved else ""} porusza sie"
    label_color = (255, 0, 0) if not rim_moved else (0, 255, 0)
    cv.putText(currentFrame, label, (x1, y1 - 20), cv.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
    
    rim_color = (255, 0, 0) if not rim_moved else (0, 255, 0)

    cv.rectangle(currentFrame, (x1, y1), (x2, y2), rim_color, 2)
