import os, glob, cv2, numpy as np, gradio as gr
from functools import lru_cache

from measure_with_marker import (
    detect_marker, marker_H, marker_box, iou, rock_size_cm, bucket,
    merge_boxes, dedupe_masks, save_calibration, load_calibration,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = "calibration.json"


def _list_detectors():
    names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))]
    return [n for n in names if not n.lower().startswith(("mobile_sam", "sam2", "sam_"))]


DETECTORS = _list_detectors()
DEFAULT_MODEL = next((m for m in DETECTORS if "yolo11" in m and "seg" not in m),
                     DETECTORS[0] if DETECTORS else "")

BUCKET_COLOR = {
    "small(<5)":    (0, 200, 0),
    "medium(5-20)": (0, 165, 255),
    "large(20+)":   (0, 0, 255),
}


@lru_cache(maxsize=6)
def _load_detector(weights):
    from ultralytics import YOLO
    return YOLO(weights)


@lru_cache(maxsize=1)
def _load_sam():
    from ultralytics import SAM
    return SAM("mobile_sam.pt")


def calibrate(image_rgb, marker_cm):
    if image_rgb is None:
        return "Please upload a photo containing the marker first."
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        pts = detect_marker(gray)
    except Exception:
        return "No ArUco marker detected. Use a clear DICT_4X4_50 marker photo."
    H, k, px_side = marker_H(pts, float(marker_cm))
    save_calibration(CALIB_PATH, H, k, marker_cm=float(marker_cm), px_side=px_side)
    return (f"Calibrated & saved to {CALIB_PATH}.\n"
            f"marker ≈{px_side:.0f}px = {marker_cm}cm (≈{px_side/float(marker_cm):.1f}px/cm)\n"
            "Now tick 'Use saved calibration' and measure marker-free photos taken "
            "from the SAME fixed camera pose.")


def measure(image_rgb, weights, marker_cm, conf, imgsz, use_sam, use_saved_calib):
    if image_rgb is None:
        return None, [], "Please upload an image first."

    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    annot = img.copy()

    if use_saved_calib:
        if not os.path.exists(CALIB_PATH):
            return image_rgb, [], "No saved calibration found. Run 'Calibrate' with a marker photo first."
        H, k = load_calibration(CALIB_PATH)
        mbox = None
        status = ["Using saved calibration (marker-free mode)."]
    else:
        try:
            pts = detect_marker(gray)
        except Exception:
            return image_rgb, [], "No ArUco marker detected. Use a clear DICT_4X4_50 marker, or tick 'Use saved calibration'."
        H, k, px_side = marker_H(pts, float(marker_cm))
        mbox = marker_box(pts)
        cv2.polylines(annot, [pts.astype(np.int32)], True, (255, 0, 0), 3)
        status = [f"marker pixel edge ~{px_side:.0f}px = {marker_cm}cm (~{px_side/float(marker_cm):.1f}px/cm)"]

    det = _load_detector(weights)
    r = det.predict(img, conf=float(conf), imgsz=int(imgsz), verbose=False)[0]
    status.append(f"model: {weights}")
    boxes = [b.xyxy.cpu().numpy().flatten() for b in r.boxes
             if mbox is None or iou(b.xyxy.cpu().numpy().flatten(), mbox) < 0.3]
    if not boxes:
        return cv2.cvtColor(annot, cv2.COLOR_BGR2RGB), [], status[0] + "\nNo rocks detected. Try lowering conf."
    raw_n = len(boxes)
    boxes = merge_boxes(boxes)

    if use_sam:
        sam = _load_sam()
        masks = sam.predict(img, bboxes=np.array(boxes), verbose=False)[0].masks.data.cpu().numpy()
        masks = dedupe_masks([(m > 0.5) for m in masks])
        items = []
        for mask in masks:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            items.append(([xs.min(), ys.min(), xs.max(), ys.max()], mask))
    else:
        items = [(box, None) for box in boxes]

    status.append(f"Detected {len(items)} rock(s)" +
                  (f" (merged from {raw_n} boxes)" if raw_n != len(items) else ""))

    rows = []
    for i, (box, mask) in enumerate(items):
        x1, y1, x2, y2 = np.asarray(box).astype(int)
        if mask is not None:
            L, S, A = rock_size_cm(mask, H, k)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cv2.drawContours(annot, [max(cnts, key=cv2.contourArea)], -1, (255, 255, 0), 2)
        else:
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            w = cv2.perspectiveTransform(corners[None], H)[0]
            (bw, bh) = cv2.minAreaRect(w)[1]
            L, S = max(bw, bh) / k, min(bw, bh) / k
            A = (bw * bh) / (k * k)

        b = bucket(L)
        color = BUCKET_COLOR[b]
        cv2.rectangle(annot, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annot, f"{i+1}:{L:.0f}cm", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        rows.append([i + 1, f"{L:.1f}", b])

    status.append("Size buckets: small <5cm / medium 5-20cm / large 20+cm")
    return cv2.cvtColor(annot, cv2.COLOR_BGR2RGB), rows, "\n".join(status)


def benchmark(imgsz, runs=10, warmup=2):
    """Time every detector on a random dummy image (speed is content-independent)."""
    from ultralytics.utils.torch_utils import get_num_params, get_flops
    dummy = (np.random.rand(1200, 1600, 3) * 255).astype(np.uint8)
    rows = []
    for m in DETECTORS:
        model = _load_detector(m)
        try:
            params = get_num_params(model.model)
            flops = get_flops(model.model, imgsz=640)
        except Exception:
            params = flops = None
        for _ in range(warmup):
            model.predict(dummy, imgsz=int(imgsz), device="cpu", verbose=False)
        ts = []
        for _ in range(runs):
            r = model.predict(dummy, imgsz=int(imgsz), device="cpu", verbose=False)[0]
            ts.append(r.speed["inference"])
        ts = np.array(ts)
        size_mb = os.path.getsize(os.path.join(MODEL_DIR, m)) / 1e6
        rows.append([
            m,
            f"{size_mb:.1f}",
            f"{params/1e6:.2f}" if params else "-",
            f"{flops:.1f}" if flops else "-",
            f"{ts.mean():.1f}±{ts.std():.1f}",
            f"{1000/ts.mean():.1f}",
        ])
    return rows


with gr.Blocks(title="Rock Size Measurement") as demo:
    gr.Markdown("## Rock Size Measurement\n"
                "Fixed-rig workflow: calibrate once with a marker photo, then measure "
                "marker-free photos from the same camera pose. Or leave calibration off "
                "and keep a marker in every photo.")

    with gr.Accordion("Calibrate once (fixed rig, marker photo)", open=False):
        gr.Markdown("Take one photo with the marker on the ground to get H + k. "
                    "Valid only while the camera pose stays fixed; re-calibrate if you "
                    "move the camera.")
        with gr.Row():
            with gr.Column(scale=1):
                calib_img = gr.Image(type="numpy", label="Marker photo")
                calib_marker_cm = gr.Number(value=10.0, label="Marker real edge length (cm)")
                calib_btn = gr.Button("Calibrate & save", variant="primary")
            calib_status = gr.Textbox(label="Calibration status", lines=4, scale=2)

    gr.Markdown("### Measure")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Input image")
            model_sel = gr.Dropdown(DETECTORS, value=DEFAULT_MODEL, label="Detection model")
            use_saved_calib = gr.Checkbox(value=False,
                label="Use saved calibration (no marker needed)")
            marker_cm = gr.Number(value=10.0, label="Marker real edge length (cm) — used only if calibration is off")
            conf = gr.Slider(0.05, 0.9, value=0.5, step=0.05, label="Detection confidence (conf)")
            imgsz = gr.Dropdown([640, 1280, 1600], value=640, label="Inference resolution (imgsz)")
            use_sam = gr.Checkbox(value=True, label="Use SAM for refined contour (more accurate, slightly slower)")
            btn = gr.Button("Measure", variant="primary")
        with gr.Column(scale=2):
            out_img = gr.Image(label="Annotated result")
            out_status = gr.Textbox(label="Status", lines=3)
            out_table = gr.Dataframe(
                headers=["#", "Longest (cm)", "Size"],
                label="Per rock", wrap=True)

    gr.Markdown("### Model comparison (speed & size)")
    gr.Markdown("Times every detector on a random image (CPU, warmup excluded). "
                "GFLOPs reported at imgsz 640.")
    with gr.Row():
        bench_imgsz = gr.Dropdown([640, 1280, 1600], value=640, label="Benchmark imgsz")
        bench_btn = gr.Button("Run benchmark", variant="secondary")
    bench_table = gr.Dataframe(
        headers=["Model", "Size (MB)", "Params (M)", "GFLOPs", "Infer ms (mean±std)", "FPS"],
        label="Lower ms / higher FPS = faster", wrap=True)

    calib_btn.click(calibrate, [calib_img, calib_marker_cm], [calib_status])
    btn.click(measure, [inp, model_sel, marker_cm, conf, imgsz, use_sam, use_saved_calib],
              [out_img, out_table, out_status])
    bench_btn.click(benchmark, [bench_imgsz], [bench_table])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
