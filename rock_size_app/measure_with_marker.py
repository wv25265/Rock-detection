import argparse, json, cv2, numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def _tuned_params():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 6
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    return p


def detect_marker(gray):
    """Detect the marker (raw -> CLAHE -> upscale) and return 4 corners TL,TR,BR,BL."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    det = cv2.aruco.ArucoDetector(d, _tuned_params())
    for stage in ("raw", "clahe", "upscale"):
        g, s = gray, 1.0
        if stage == "clahe":
            g = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
        elif stage == "upscale":
            s = 1.5
            g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = det.detectMarkers(g)
        if ids is not None:
            return corners[0].reshape(4, 2).astype(np.float32) / s
    raise RuntimeError("No ArUco marker detected")


def marker_box(pts):
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def marker_H(pts, marker_cm, px_per_cm=40.0):
    """Return (H, k, mean_edge_px). H maps source pixels to a fronto-parallel plane
    where k = px_per_cm."""
    side = marker_cm * px_per_cm
    dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
    H, _ = cv2.findHomography(pts, dst)
    e = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return H, px_per_cm, float(np.mean(e))


def save_calibration(path, H, k, marker_cm=None, px_side=None):
    """Save H + k so a fixed-rig camera can measure marker-free photos later."""
    d = {"H": np.asarray(H).tolist(), "k": float(k)}
    if marker_cm is not None:
        d["marker_cm"] = float(marker_cm)
    if px_side is not None:
        d["px_side"] = float(px_side)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


def load_calibration(path):
    with open(path) as f:
        d = json.load(f)
    return np.array(d["H"], dtype=np.float64), float(d["k"])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def _feret_max(pts):
    """Max caliper diameter: largest distance between any two hull points."""
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 2:
        return 0.0
    d = 0.0
    for i in range(len(hull)):
        diff = hull - hull[i]
        m = float(np.sqrt((diff ** 2).sum(1)).max())
        if m > d:
            d = m
    return d


def _containment(a, b):
    """intersection over the smaller box area; catches nested duplicates."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    m = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return inter / m if m > 0 else 0.0


def merge_boxes(boxes, iou_thr=0.3, contain_thr=0.4):
    """Merge overlapping/contained boxes (union bbox) so one rock split across
    several boxes becomes one. Iterates until stable."""
    boxes = [np.asarray(b, dtype=float) for b in boxes]
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        out, skip = [], set()
        for i in range(len(boxes)):
            if i in skip:
                continue
            a = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in skip:
                    continue
                b = boxes[j]
                if iou(a, b) >= iou_thr or _containment(a, b) >= contain_thr:
                    a = np.array([min(a[0], b[0]), min(a[1], b[1]),
                                  max(a[2], b[2]), max(a[3], b[3])])
                    skip.add(j)
                    changed = True
            out.append(a)
        boxes = out
    return boxes


def dedupe_masks(masks, thr=0.5):
    """Merge SAM masks belonging to the same rock. Returns uint8 masks."""
    masks = [np.asarray(m).astype(bool) for m in masks]
    used, out = [False] * len(masks), []
    for i in range(len(masks)):
        if used[i]:
            continue
        m = masks[i].copy()
        for j in range(i + 1, len(masks)):
            if used[j]:
                continue
            inter = np.logical_and(m, masks[j]).sum()
            smaller = min(int(m.sum()), int(masks[j].sum()))
            if smaller > 0 and inter / smaller > thr:
                m = np.logical_or(m, masks[j])
                used[j] = True
        out.append(m.astype(np.uint8))
        used[i] = True
    return out


def rock_size_cm(mask, H, k):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(c[None], H)[0]
    (w, h) = cv2.minAreaRect(warped)[1]
    L = _feret_max(warped) / k
    return L, min(w, h) / k, cv2.contourArea(warped) / (k * k)


def bucket(L):
    return "small(<5)" if L < 5 else "medium(5-20)" if L < 20 else "large(20+)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--marker_cm", type=float, required=True)
    ap.add_argument("--weights", default="rock_yolo11.s_best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1600)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = detect_marker(gray)
    H, k, px_side = marker_H(pts, args.marker_cm)
    mbox = marker_box(pts)
    print(f"[marker] pixel edge≈{px_side:.1f}px real={args.marker_cm}cm "
          f"-> source scale≈{px_side/args.marker_cm:.2f}px/cm; H computed")

    try:
        from ultralytics import YOLO, SAM
    except ImportError:
        print("[rock] ultralytics not available, skipping rock measurement")
        return
    det = YOLO(args.weights)
    r = det.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    boxes = [b.xyxy.cpu().numpy().flatten() for b in r.boxes
             if iou(b.xyxy.cpu().numpy().flatten(), mbox) < 0.3]
    if not boxes:
        print("[rock] no rocks detected (marker excluded)")
        return
    sam = SAM("mobile_sam.pt")
    masks = sam.predict(img, bboxes=np.array(boxes), verbose=False)[0].masks.data.cpu().numpy()
    for i, m in enumerate(masks):
        L, S, A = rock_size_cm((m > 0.5).astype(np.uint8), H, k)
        print(f"[rock {i+1}] longest={L:.1f}cm shortest={S:.1f}cm area={A:.1f}cm2 -> {bucket(L)}")


if __name__ == "__main__":
    main()
