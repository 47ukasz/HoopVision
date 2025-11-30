import cv2 as cv
import numpy as np

from constants import BASKETBALL_CLASS_INDEX, MIN_BALL_CONF

def handle_detect_trajectory_point(currentFrameId, box):
    box_class_index = int(box.cls[0])
    box_conf_level = float(box.conf[0])

    if box_class_index != BASKETBALL_CLASS_INDEX:
        return None
    
    if box_conf_level < MIN_BALL_CONF:
        return None
    
    xyxy = box.xyxy[0].cpu().numpy()

    center_x = int((xyxy[0] + xyxy[2]) / 2)
    center_y = int((xyxy[1] + xyxy[3]) / 2)

    point = (currentFrameId, center_x, center_y)

    return point

def handle_draw_trajectory(currentFrame, trajectory_points, color = (255,0,0), lineWidth = 5):
    trajectory_size = len(trajectory_points)

    if trajectory_size == 0:
        return
    
    for index in range(1, trajectory_size):
        start_point = trajectory_points[index-1][1:]
        end_point = trajectory_points[index][1:]
        cv.line(currentFrame, start_point, end_point, color, lineWidth)