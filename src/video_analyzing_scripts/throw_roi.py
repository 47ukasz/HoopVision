def create_tracking_roi(rim_box, frame_shape, margin_xmin = 150, margin_ymax = 50, margin_xmax = 600, margin_ymin = 300):
    """
    Tworzymy ROI poprzez powiekszneie obszaru RIM
    """
    if rim_box is None:
        return None
        
    h, w = frame_shape[:2]
    r_xmin, r_ymin, r_xmax, r_ymax = rim_box

    # dodanie marginesow do boxa rim
    roi_xmin = max(0, int(r_xmin - margin_xmin))
    roi_ymin = max(0, int(r_ymin - margin_ymin))
    roi_xmax = min(w, int(r_xmax + margin_xmax))
    roi_ymax = min(h, int(r_ymax + margin_ymax))

    return [roi_xmin, roi_ymin, roi_xmax, roi_ymax]

def is_ball_in_roi(ball_box, roi_box):
    """
    Sprawdza czy srodek pilki w ROI
    """
    if ball_box is None or roi_box is None:
        return False

    #srodek pilki
    b_x_center = (ball_box[0] + ball_box[2]) / 2
    b_y_center = (ball_box[1] + ball_box[3]) / 2

    #czy srodek pilki w ROI
    if (roi_box[0] <= b_x_center <= roi_box[2] and 
        roi_box[1] <= b_y_center <= roi_box[3]):
        return True
    return False