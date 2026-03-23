import io
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, send_file, Response, jsonify

# =========================
# Config (env-tweakable)
# =========================
MODELS_DIR = Path("models")

# Class IDs
HEAD_ID   = int(os.getenv("HEAD_ID", "2"))
HELMET_ID = int(os.getenv("HELMET_ID", "3"))
PERSON_ID = int(os.getenv("PERSON_ID", "1"))

# Class-specific thresholds
THR_PERSON = float(os.getenv("THR_PERSON", "0.10"))
THR_HEAD   = float(os.getenv("THR_HEAD",   "0.30"))   # relaxed to catch heads
THR_HELMET = float(os.getenv("THR_HELMET", "0.75"))   # stricter for fewer FPs

# Webcam index
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))

# Separate sizes (speed vs. accuracy)
IMG_SIZE_CAM    = int(os.getenv("IMG_SIZE_CAM", "320"))    # webcam speed
IMG_SIZE_UPLOAD = int(os.getenv("IMG_SIZE_UPLOAD", "640")) # upload accuracy

# Torch FP16 speedup (CUDA only)
TORCH_FP16 = os.getenv("TORCH_FP16", "1") == "1" and torch.cuda.is_available()

# Prefer state_dict (.pth) first
PREFER_STATE = os.getenv("PREFER_STATE", "1")

# Stricter helmet-head matching (uploads now strict too)
STRICT_MODE = os.getenv("STRICT_MODE", "1") == "1"
TOP_FRAC = float(os.getenv("TOP_FRAC", "0.6"))                  # helmet bottom must be in top 60% of head
MIN_HELM_OVERLAP = float(os.getenv("MIN_HELM_OVERLAP", "0.40")) # min (helmet∩head)/helmet
SHOW_HELMETS = os.getenv("SHOW_HELMETS", "0") == "1"            # debug: draw helmet boxes
STRICT_HELMET_SCORE = float(os.getenv("STRICT_HELMET_SCORE", "0.85"))

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type: Optional[str] = None
model = None


# =========================
# Geometry helpers
# =========================
def to_xyxy(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1, y1, x2, y2

def area(b):
    x1, y1, x2, y2 = to_xyxy(b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def inter_area(a, b):
    ax1, ay1, ax2, ay2 = to_xyxy(a); bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

def iou(a, b) -> float:
    ia = inter_area(a, b)
    if ia <= 0: return 0.0
    return ia / max(1e-6, area(a) + area(b) - ia)

def head_is_covered(
    head_box,
    helmet_boxes,
    helmet_scores=None,
    *,
    iou_thresh=0.25,
    top_frac=0.6,
    min_inter_frac=0.35,
    min_helmet_score=0.85,
    strict=True,
    min_area_ratio=0.15,  # helmet area must be >= 15% of head area
    max_area_ratio=0.75,  # helmet area must be <= 75% of head area
    center_top_frac=0.55  # helmet center must be in top 55% of head
) -> bool:
    """
    Green only if a plausible helmet sits on the upper part of the head,
    with reasonable size and placement.
    """
    hx1, hy1, hx2, hy2 = to_xyxy(head_box)
    h_w = max(1.0, hx2 - hx1)
    h_h = max(1.0, hy2 - hy1)
    head_area = h_w * h_h
    top_limit = hy1 + top_frac * h_h

    for idx, hb in enumerate(helmet_boxes):
        if helmet_scores is not None and strict and helmet_scores[idx] < min_helmet_score:
            continue

        bx1, by1, bx2, by2 = to_xyxy(hb)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        helm_area = bw * bh

        # Size sanity vs head
        area_ratio = helm_area / head_area
        if strict and not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        # Basic overlap
        ov_iou = iou(head_box, hb)
        if ov_iou < iou_thresh:
            continue

        # Helmet must be in upper part of head
        if strict and by2 > top_limit:
            continue

        # Enough of helmet actually overlapping head
        frac = inter_area(head_box, hb) / max(1e-6, helm_area)
        if strict and frac < min_inter_frac:
            continue

        # Helmet center inside head and upper portion
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        if strict:
            if not (hx1 <= cx <= hx2):       # center x inside head
                continue
            if cy > (hy1 + center_top_frac * h_h):
                continue

        return True
    return False


# =========================
# Letterbox
# =========================
def letterbox_bgr(img_bgr, size=416, fill=(114,114,114)):
    h0, w0 = img_bgr.shape[:2]
    scale = min(size / w0, size / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    im_resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - nw)//2, (size - nh)//2
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    canvas[dy:dy+nh, dx:dx+nw] = im_resized

    def map_back(boxes416: np.ndarray):
        if boxes416.size == 0: return boxes416
        b = boxes416.astype(np.float32).copy()
        b[:, [0,2]] -= dx; b[:, [1,3]] -= dy
        b /= scale
        b[:,0] = np.clip(b[:,0], 0, w0-1); b[:,2] = np.clip(b[:,2], 0, w0-1)
        b[:,1] = np.clip(b[:,1], 0, h0-1); b[:,3] = np.clip(b[:,3], 0, h0-1)
        return b

    return canvas, map_back


# =========================
# Model loading
# =========================
def find_models() -> Tuple[Optional[str], Optional[Path]]:
    prefer_state = PREFER_STATE == "1"
    candidates = []
    if prefer_state:
        candidates = [("state", MODELS_DIR / "best_model.pth"),
                      ("state", MODELS_DIR / "final_model.pth"),
                      ("scripted", MODELS_DIR / "fasterrcnn_scripted.pt")]
    else:
        candidates = [("scripted", MODELS_DIR / "fasterrcnn_scripted.pt"),
                      ("state", MODELS_DIR / "best_model.pth"),
                      ("state", MODELS_DIR / "final_model.pth")]
    for t, p in candidates:
        if p.exists():
            return t, p
    return None, None

def build_state_model():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=4)

def load_model():
    global model, model_type
    mtype, path = find_models()
    if mtype is None:
        print("No model found in ./models.")
        return
    if mtype == "scripted":
        try:
            m = torch.jit.load(str(path), map_location=device).eval().to(device)
            model, model_type = m, "scripted"
            print(f"Loaded TorchScript model: {path}")
            return
        except Exception as e:
            print("TorchScript load failed:", e)
    for alt in ("best_model.pth", "final_model.pth"):
        p = MODELS_DIR / alt
        if p.exists():
            m = build_state_model().to(device).eval()
            try:
                state = torch.load(str(p), map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(str(p), map_location=device)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            m.load_state_dict(state, strict=False)
            m.eval()
            model, model_type = m, "state"
            print(f"Loaded state_dict model: {p}")
            return
    raise RuntimeError("Could not load any model.")


# =========================
# Inference
# =========================
def infer_numpy(frame_bgr, debug=False, img_size=416, strict_override=None, draw_helmet_fallback=False):
    if model is None:
        load_model()
        if model is None:
            raise RuntimeError("Model not loaded.")

    lb_img, map_back = letterbox_bgr(frame_bgr, size=img_size)
    rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).float() / 255.0
    t = t.permute(2,0,1).unsqueeze(0)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=TORCH_FP16):
        if model_type == "scripted":
            out = model(t)
            if isinstance(out, (list, tuple)) and len(out) == 3:
                boxes = out[0].detach().cpu().numpy()
                labels = out[1].detach().cpu().numpy()
                scores = out[2].detach().cpu().numpy()
            else:
                det = out[0] if isinstance(out, (list,tuple)) else out
                boxes  = det["boxes"].detach().cpu().numpy()
                labels = det["labels"].detach().cpu().numpy()
                scores = det["scores"].detach().cpu().numpy()
        else:
            det = model(t)[0]
            boxes  = det["boxes"].detach().cpu().numpy()
            labels = det["labels"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

    boxes = map_back(boxes)

    # Class-specific thresholds
    keep = []
    for l, s in zip(labels, scores):
        if   int(l) == PERSON_ID: keep.append(s >= THR_PERSON)
        elif int(l) == HEAD_ID:   keep.append(s >= THR_HEAD)
        elif int(l) == HELMET_ID: keep.append(s >= THR_HELMET)
        else:                     keep.append(False)
    keep = np.array(keep, dtype=bool)

    boxes_kept  = boxes[keep]
    labels_kept = labels[keep]
    scores_kept = scores[keep]

    # Helmet collections
    helmet_boxes  = [boxes_kept[i]  for i, lab in enumerate(labels_kept) if int(lab) == HELMET_ID]
    helmet_scores = [scores_kept[i] for i, lab in enumerate(labels_kept) if int(lab) == HELMET_ID]

    if SHOW_HELMETS:
        for hb in helmet_boxes:
            x1,y1,x2,y2 = map(int, hb)
            cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (255,0,0), 1)

    # Draw ONLY heads; color by helmet coverage
    heads_drawn = 0
    for i, lab in enumerate(labels_kept):
        if int(lab) != HEAD_ID: 
            continue
        x1, y1, x2, y2 = map(int, boxes_kept[i])
        covered = head_is_covered(
            boxes_kept[i], helmet_boxes, helmet_scores,
            iou_thresh=0.25,
            top_frac=TOP_FRAC,
            min_inter_frac=MIN_HELM_OVERLAP,
            min_helmet_score=STRICT_HELMET_SCORE,
            strict=STRICT_MODE if strict_override is None else strict_override
        )
        color = (0,200,0) if covered else (0,0,255)
        cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame_bgr, "Helmet Worn" if covered else "No Helmet",
                    (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        heads_drawn += 1

    # Fallback: for uploads, if no heads drawn but helmets exist, draw helmets so user sees something
    if draw_helmet_fallback and heads_drawn == 0 and len(helmet_boxes) > 0:
        for hb in helmet_boxes:
            x1, y1, x2, y2 = map(int, hb)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 200), 2)
            cv2.putText(frame_bgr, "Helmet", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 2, cv2.LINE_AA)

    if debug:
        print(f"[DEBUG] heads_drawn={heads_drawn}, helmets_found={len(helmet_boxes)}")
    return frame_bgr


# =========================
# Flask routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/infer_upload", methods=["POST"])
def infer_upload():
    if "image" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Uploads: accuracy-focused size and STRICT matching to avoid false greens
    vis = infer_numpy(
        frame,
        debug=True,
        img_size=IMG_SIZE_UPLOAD,
        strict_override=True,            # strict on uploads, too
        draw_helmet_fallback=True
    )
    ok, enc = cv2.imencode(".jpg", vis)
    return send_file(io.BytesIO(enc.tobytes()), mimetype="image/jpeg")

def gen_frames():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    while True:
        ok, frame = cap.read()
        if not ok: break
        # Webcam: speed-focused size, strict to avoid false greens
        vis = infer_numpy(
            frame,
            debug=False,
            img_size=IMG_SIZE_CAM,
            strict_override=True,
            draw_helmet_fallback=False
        )
        ok, buffer = cv2.imencode(".jpg", vis)
        if not ok: continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
