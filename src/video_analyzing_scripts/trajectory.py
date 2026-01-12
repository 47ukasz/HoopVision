import cv2 as cv
import numpy as np

from constants import BASKETBALL_CLASS_INDEX, MIN_BALL_CONF

def handle_detect_trajectory_point(currentFrameId, box):
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


def handle_draw_ball(frame, b_xyxy, conf=None, color=(0, 255, 255)):
    if b_xyxy is None:
        return

    x1, y1, x2, y2 = map(int, b_xyxy)

    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cx = int((x1 + x2) * 0.5)
    cy = int((y1 + y2) * 0.5)
    cv.circle(frame, (cx, cy), 3, color, -1)

    label = "BALL"
    if conf is not None:
        label += f" {conf:.2f}"
    cv.putText(frame, label, (x1, max(0, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
