# Rock Size Estimation — Formulas

Detection is done by YOLOv8 (deep learning); **physical size is estimated by camera
geometry (pinhole model), not machine learning.** This document lists the formulas.

---

## Notation

| Symbol | Meaning | Unit |
|--------|---------|------|
| $S$ | real-world size of the rock (longest side) | cm |
| $p$ | bounding-box size in the image (longest side) | pixels |
| $D$ | camera-to-object distance (shooting height) | cm |
| $f_{px}$ | focal length expressed in pixels | pixels |
| $f_{35}$ | 35 mm-equivalent focal length (from EXIF) | mm |
| $L$ | image long-edge resolution | pixels |
| $k$ | scale factor (pixels per cm) | px/cm |
| $D_i$ | per-pixel depth at box centre (LiDAR) | cm |

---

## 1. Pinhole camera model (basis)

$$\frac{S}{D} = \frac{p}{f_{px}} \quad\Longrightarrow\quad S = \frac{p \cdot D}{f_{px}}$$

## 2. Focal length in pixels (from EXIF)

$$f_{px} = \frac{f_{35} \cdot L}{36}$$

36 mm = long edge of a full-frame 35 mm sensor.
iPhone 12: $f_{px} = \dfrac{26 \times 4032}{36} = 2912$ px.
iPhone 15 Pro: $f_{px} = \dfrac{24 \times 5712}{36} = 3808$ px.

## 3. Size from fixed height (Method A — focal)

$$\boxed{\,S = \frac{p \cdot D}{f_{px}} = \frac{36\, p\, D}{f_{35}\, L}\,}$$

## 4. Size from ruler calibration (Method B — recommended)

Calibrate once at a fixed height by photographing a ruler:

$$k = \frac{p_{ref}}{S_{ref}} \qquad (\text{pixels between two marks} / \text{known cm})$$

Then for each detected rock:

$$\boxed{\,S = \frac{p}{k}\,}$$

Methods A and B are equivalent, since $k = f_{px} / D$.

## 5. Size from LiDAR depth (Method C — iPhone 15 Pro, distance may vary)

Using the per-pixel depth $D_i$ at the box centre:

$$S = \frac{p \cdot D_i}{f_{px}}$$

No fixed height required, because $D_i$ is measured directly for each object.

## 6. Bounding-box pixel size

For a YOLO box $(x_1, y_1, x_2, y_2)$:

$$p = \max\big(x_2 - x_1,\; y_2 - y_1\big)$$

## 7. Size-class assignment

$$
\text{class}(S) =
\begin{cases}
\text{small} & 5 \le S < 10 \\
\text{medium} & 10 \le S < 20 \\
\text{large} & S \ge 20
\end{cases}
$$

## 8. Accuracy validation (report this)

Mean Absolute Error against caliper ground truth over $N$ rocks:

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} \big| S_i^{\text{pred}} - S_i^{\text{true}} \big| \quad (\text{cm})$$

---

## Assumptions & error sources

- Camera held **perpendicular (top-down)** to the ground plane.
- Same camera lens (main 1×), same resolution, **no zoom** → $f_{px}$ constant.
- Rock height introduces a small bias ($D_{\text{effective}} = D - h_{\text{rock}}$).
- Methods A/B require a **known/fixed shooting height**; Method C does not.
