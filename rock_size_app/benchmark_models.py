"""Benchmark detectors on speed, size, params and GFLOPs.

Speed depends on input size, not image content, so it defaults to a random dummy
image. Pass --images to time on real photos instead.

  python benchmark_models.py
  python benchmark_models.py --device mps
  python benchmark_models.py --imgsz 640 --runs 30
  python benchmark_models.py --images ./some_dir
"""
import argparse, os, glob
import numpy as np
from ultralytics import YOLO

DEFAULT_MODELS = [
    "rock_yolo11.n_best.pt",
    "rock_yolo11.s_best.pt",
    "rock_yolo8.s_best.pt",
    "yolo11n-seg.pt",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="model paths (default: folder set)")
    ap.add_argument("--imgsz", type=int, default=1600)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="cpu / mps / 0 (cuda)")
    ap.add_argument("--images", default=None, help="dir/file of real images to time on")
    args = ap.parse_args()

    from ultralytics.utils.torch_utils import get_num_params, get_flops
    models = args.models or [m for m in DEFAULT_MODELS if os.path.exists(m)]

    if args.images:
        import cv2
        paths = ([args.images] if os.path.isfile(args.images)
                 else sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                             glob.glob(os.path.join(args.images, "*.png"))))
        srcs = [cv2.imread(p) for p in paths[:args.runs]] or None
        if not srcs:
            raise SystemExit(f"No images found in {args.images}")
        src_desc = f"{len(srcs)} real image(s)"
    else:
        srcs = [(np.random.rand(1200, 1600, 3) * 255).astype(np.uint8)]
        src_desc = "1 random dummy image"

    print(f"device={args.device}  imgsz={args.imgsz}  runs={args.runs}  input={src_desc}\n")
    header = f"{'model':26s} {'size':>8s} {'params':>9s} {'GFLOPs':>8s} {'infer ms':>14s} {'FPS':>7s}"
    print(header)
    print("-" * len(header))

    for m in models:
        model = YOLO(m)
        try:
            params = get_num_params(model.model)
            flops = get_flops(model.model, imgsz=640)
        except Exception:
            params = flops = None

        for i in range(args.warmup):
            model.predict(srcs[i % len(srcs)], imgsz=args.imgsz,
                          device=args.device, verbose=False)

        times = []
        for i in range(args.runs):
            r = model.predict(srcs[i % len(srcs)], imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
            times.append(r.speed["inference"])
        t = np.array(times)

        size_mb = os.path.getsize(m) / 1e6
        pstr = f"{params/1e6:.2f}M" if params else "  -"
        fstr = f"{flops:.1f}" if flops else "  -"
        print(f"{os.path.basename(m):26s} {size_mb:6.1f}MB {pstr:>9s} {fstr:>8s} "
              f"{t.mean():7.1f}±{t.std():4.1f} {1000/t.mean():7.1f}")

    print("\nnotes: infer ms = pure inference (excludes pre/post-processing); "
          "GFLOPs reported at imgsz 640; run with --device mps for Apple GPU.")


if __name__ == "__main__":
    main()
