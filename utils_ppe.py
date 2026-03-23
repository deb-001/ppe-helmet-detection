
import numpy as np

ID_TO_CLASS = {1: "person", 2: "head", 3: "helmet"}

def to_xyxy(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1, y1, x2, y2

def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, (area_a + area_b - inter))

def center_in_box(head_box, helm_box) -> bool:
    hx1, hy1, hx2, hy2 = to_xyxy(head_box)
    cx, cy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
    bx1, by1, bx2, by2 = to_xyxy(helm_box)
    return (bx1 <= cx <= bx2) and (by1 <= cy <= by2)

def match_head_to_helmet(head_box, helmet_boxes, iou_thresh=0.3) -> bool:
    for hb in helmet_boxes:
        if center_in_box(head_box, hb) or iou(head_box, hb) >= iou_thresh:
            return True
    return False
