"""
Phase 9a prototype: detect individual red blood cells in a wide-field smear photo.

Pipeline (OpenCV): work at ~1000px longest side -> green channel + CLAHE (RBCs are
darker than background in green, for both the pink and purple stains in this data)
-> field mask (drop the dark microscope surround, erode the rim) -> Otsu or adaptive
threshold inside the field -> morphology (open/close, fill central-pallor holes) ->
distance-transform + watershed to split touching cells -> per-blob filtering.

There is NO ground truth for cell positions in these datasets, so detection quality
is judged from area statistics (proxies for merges/fragments) and by eye on overlays;
see the Phase 9a report. Nothing here is Android code.
"""
import cv2
import numpy as np

WORK_SIDE = 1000


def load_bgr(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def field_mask(gray_small):
    """Bright circular microscope field vs dark surround. Returns uint8 mask (255 = field)."""
    blur = cv2.GaussianBlur(gray_small, (0, 0), 5)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (blur > t * 0.6).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.full_like(gray_small, 255)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == biggest).astype(np.uint8) * 255
    # fill any holes, then use the convex hull (the field is round)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    out = np.zeros_like(m)
    cv2.drawContours(out, [hull], -1, 255, -1)
    return out


def foreground(green_small, fmask, method="otsu"):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(green_small)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    inv = 255 - g
    if method == "adaptive":
        # local threshold: robust to uneven illumination across the field
        fg = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -6)
    else:
        vals = inv[fmask > 0]
        t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = (inv > t).astype(np.uint8) * 255
    fg = cv2.bitwise_and(fg, fmask)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k5)
    # fill central-pallor holes (small enclosed background inside a cell)
    inv_fg = cv2.bitwise_not(fg)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv_fg)
    holes = np.zeros_like(fg)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        touches_border = x == 0 or y == 0 or x + w == fg.shape[1] or y + h == fg.shape[0]
        if not touches_border and a < 900:
            holes[lab == i] = 255
    return cv2.bitwise_or(fg, holes)


def estimate_radius(fg):
    """Typical single-cell radius: median equivalent radius of compact, mid-sized blobs."""
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    radii = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 150:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull > 0 and a / hull > 0.94:
            radii.append(np.sqrt(a / np.pi))
    return float(np.median(radii)) if len(radii) >= 5 else 18.0


def split_touching(fg, r_est, color_small):
    """Distance transform + watershed; markers = local maxima of the smoothed distance map."""
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    dist_s = cv2.GaussianBlur(dist, (0, 0), 2.0)
    k = max(3, int(r_est * 1.1) | 1)
    local_max = (dist_s == cv2.dilate(dist_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))) & (dist_s > 0.45 * r_est)
    n_m, markers = cv2.connectedComponents(local_max.astype(np.uint8))
    markers = markers + 1                      # 1 = background label, cells start at 2
    markers[fg == 0] = 1
    unknown = (fg > 0) & (markers == 1)        # foreground pixels not yet claimed by a marker
    markers[unknown] = 0
    # flood on the (lightly blurred) original colour image so boundaries follow real cell edges
    vis = cv2.GaussianBlur(color_small, (0, 0), 1.2)
    markers = cv2.watershed(vis, markers.astype(np.int32))
    markers[markers <= 1] = 0
    return markers


def detect_cells(img_bgr, method="adaptive"):
    """Returns dict: cells (list of dicts, full-res coords), plus intermediates for plotting."""
    h, w = img_bgr.shape[:2]
    scale = WORK_SIDE / max(h, w)
    small = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    fmask = field_mask(gray)
    # ignore a rim of the field: cells cut by the field edge can't be classified as whole cells
    rim = max(6, int(0.02 * max(small.shape[:2])))
    fmask_in = cv2.erode(fmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1, 2 * rim + 1)))
    fg = foreground(small[:, :, 1], fmask, method)
    r_est = estimate_radius(fg)
    labels = split_touching(fg, r_est, small)

    raw = []
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = (labels == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < 30:
            continue
        per = cv2.arcLength(c, True)
        hull_a = cv2.contourArea(cv2.convexHull(c))
        x, y, bw, bh = cv2.boundingRect(c)
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        raw.append(dict(
            label=int(lab), area=float(a), circularity=float(4 * np.pi * a / (per * per + 1e-9)),
            solidity=float(a / (hull_a + 1e-9)), bbox=(x, y, bw, bh), center=(float(cx), float(cy)),
            touches_edge=bool(fmask_in[min(int(cy), fmask_in.shape[0] - 1), min(int(cx), fmask_in.shape[1] - 1)] == 0),
            mask=m,
        ))

    areas = np.array([r["area"] for r in raw]) if raw else np.array([1.0])
    med = float(np.median(areas))
    for r in raw:
        r["rel_area"] = r["area"] / med
        # keep: mid-sized, not cut by the field edge. Merged clumps (>2x median) and tiny debris are dropped.
        r["kept"] = (0.45 <= r["rel_area"] <= 2.0) and not r["touches_edge"] and r["solidity"] > 0.80

    return dict(scale=scale, small=small, fmask=fmask, fg=fg, labels=labels, cells=raw, median_area=med, r_est=r_est)


def hough_count(img_bgr, r_est_small):
    """Comparison detector: Hough circles at the estimated cell radius (working scale)."""
    h, w = img_bgr.shape[:2]
    scale = WORK_SIDE / max(h, w)
    small = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 2)
    r = max(6, int(r_est_small))
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(r * 1.3), param1=100, param2=22,
                         minRadius=int(r * 0.7), maxRadius=int(r * 1.35))
    fm = field_mask(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    if c is None:
        return 0
    return int(sum(1 for x, y, _ in c[0] if fm[int(np.clip(y, 0, fm.shape[0] - 1)), int(np.clip(x, 0, fm.shape[1] - 1))] > 0))


def overlay(det, path):
    vis = det["small"].copy()
    fm = cv2.morphologyEx(det["fmask"], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    vis[fm > 0] = (255, 255, 0)
    for r in det["cells"]:
        cnts, _ = cv2.findContours(r["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if r["kept"]:
            color = (0, 200, 0)                 # green: kept cell
        elif r["rel_area"] > 2.0:
            color = (0, 0, 255)                 # red: too large -> probable merge / clump
        elif r["rel_area"] < 0.45:
            color = (255, 0, 0)                 # blue: too small -> fragment / debris
        else:
            color = (0, 165, 255)               # orange: edge-cut or non-convex
        cv2.drawContours(vis, cnts, -1, color, 1)
    cv2.imwrite(path, vis)
