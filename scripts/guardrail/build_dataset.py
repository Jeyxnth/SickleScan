"""
Builds the Phase 7 guardrail dataset: "is this a blood smear microscopy
image?" (smear vs not_smear).

Positive class ("smear"): reuses the existing sickle cell and malaria
images, BOTH disease labels count (the guardrail doesn't care about disease
status). Sickle cell contributes every usable image (569); malaria is
subsampled (stratified parasitized/uninfected) to keep the two smear types
from being 98% malaria -- malaria has 27,558 images vs sickle cell's 569, and
an unsubsampled positive class would barely see a sickle-cell-style field
photo, which is exactly the real-smear case the app must not wrongly reject.

Negative class ("not_smear"): diverse non-smear images from public sources
that need no account -- see NEGATIVE_SOURCES below. Deliberately NOT
CIFAR-100: its 32x32 images upscaled to 224x224 are extremely blurry, so a
classifier trained on them can learn "blurry = not a smear" instead of
"content isn't a smear" and would then wave through any sharp phone photo of
a random object (the dangerous failure mode).

Extracted negatives are downscaled to max side 512 and saved as JPEG under
data/guardrail/negatives/<source>/. Output: data/guardrail/manifest.csv with
columns path,label,source,split (70/15/15 stratified by source, seed 42).
"""
import csv
import io
import random
import tarfile
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

SEED = 42
ROOT = Path("data/guardrail")
RAW = ROOT / "raw"
OUT = ROOT / "negatives"
MAX_SIDE = 512

# Negative-class targets per source (images).
NEGATIVE_SOURCES = {
    "coco": 1000,         # everyday objects, people, indoor/outdoor scenes
    "places": 700,        # scenes / landscapes / rooms / streets
    "dtd": 450,           # textures & patterns (incl. pink/purple stained-looking ones)
    "lfw": 400,           # faces (one image per identity, for diversity)
    "screenshots": 400,   # phone/desktop/web UI screenshots
    "documents": 199,     # scanned forms / text documents (all of FUNSD)
}  # + SYNTH_COUNTS synthetic capture failures, generated below (kept as separate sources)
MALARIA_POSITIVES = 2580  # 1290 parasitized + 1290 uninfected -> smear total == negative total


def save_jpeg(img: Image.Image, dest: Path):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=90)


def from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


USED_COCO = set()


def extract_coco(rng, n):
    with zipfile.ZipFile(RAW / "coco_val2017.zip") as z:
        names = sorted(x for x in z.namelist() if x.lower().endswith(".jpg"))
        picks = rng.sample(names, n)
        USED_COCO.update(picks)
        out = []
        for i, name in enumerate(picks):
            dest = OUT / "coco" / f"{i:04d}.jpg"
            save_jpeg(Image.open(z.open(name)), dest)
            out.append(dest)
    return out


def extract_places(rng, n):
    with tarfile.open(RAW / "places_val256.tar") as t:
        members = [m for m in t.getmembers() if m.name.lower().endswith(".jpg")]
        members.sort(key=lambda m: m.name)
        picks = rng.sample(members, n)
        out = []
        for i, m in enumerate(picks):
            dest = OUT / "places" / f"{i:04d}.jpg"
            save_jpeg(Image.open(t.extractfile(m)), dest)
            out.append(dest)
    return out


def extract_dtd(rng, n):
    with tarfile.open(RAW / "dtd.tar.gz") as t:
        members = [m for m in t.getmembers() if m.name.lower().endswith(".jpg") and "/images/" in m.name]
        members.sort(key=lambda m: m.name)
        picks = rng.sample(members, n)
        out = []
        for i, m in enumerate(picks):
            dest = OUT / "dtd" / f"{i:04d}.jpg"
            save_jpeg(Image.open(t.extractfile(m)), dest)
            out.append(dest)
    return out


def extract_lfw(rng, n):
    t = pq.read_table(RAW / "lfw.parquet").to_pydict()
    by_identity = {}
    for label, img in zip(t["label"], t["image"]):
        by_identity.setdefault(label, []).append(img["bytes"])
    identities = sorted(by_identity)
    rng.shuffle(identities)
    out = []
    for i, ident in enumerate(identities[:n]):
        dest = OUT / "lfw" / f"{i:04d}.jpg"
        save_jpeg(from_bytes(rng.choice(by_identity[ident])), dest)
        out.append(dest)
    return out


def extract_screenshots(rng, n):
    rows = []
    for i in range(3):
        t = pq.read_table(RAW / f"screenspot_{i}.parquet").to_pydict()
        rows += [img["bytes"] for img in t["image"]]
    picks = rng.sample(rows, n)
    out = []
    for i, b in enumerate(picks):
        dest = OUT / "screenshots" / f"{i:04d}.jpg"
        save_jpeg(from_bytes(b), dest)
        out.append(dest)
    return out


def extract_documents(rng, n):
    rows = []
    for s in ("train", "test"):
        t = pq.read_table(RAW / f"funsd_{s}.parquet").to_pydict()
        rows += [img["bytes"] for img in t["image"]]
    picks = rows if n >= len(rows) else rng.sample(rows, n)
    out = []
    for i, b in enumerate(picks):
        dest = OUT / "documents" / f"{i:04d}.jpg"
        save_jpeg(from_bytes(b), dest)
        out.append(dest)
    return out


# ---------------------------------------------------------------------------
# Synthetic capture failures (Phase 7 follow-up). NOT real photos: generated
# procedurally to cover the most common bad real-world captures (black frame,
# blank/overexposed frame, out-of-focus shot, finger over the lens) that no
# public dataset contains. Kept as their own source labels ("synthetic_*") so
# the report can track them separately from the real-photo negatives.
# ---------------------------------------------------------------------------
import numpy as np
from PIL import ImageFilter

SYNTH_COUNTS = {
    "synthetic_black": 60,
    "synthetic_blank": 60,
    "synthetic_blur": 100,
    "synthetic_finger": 80,
}
SYNTH_SIZES = [(640, 480), (480, 640), (512, 384), (384, 512), (512, 288), (256, 256), (160, 139)]
SKIN_TONES = [(224, 172, 150), (198, 134, 108), (160, 106, 84), (120, 76, 58), (92, 58, 44), (236, 188, 166)]


def _noise(nrng, h, w, sigma):
    return nrng.normal(0, sigma, (h, w, 1))


def synth_black(nrng, size):
    w, h = size
    level = nrng.uniform(0, 22)
    img = np.full((h, w, 3), level) + _noise(nrng, h, w, nrng.uniform(0.5, 6))
    if nrng.random() < 0.4:  # faint light leak / vignette glow
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = nrng.uniform(0, w), nrng.uniform(0, h)
        glow = np.exp(-(((xx - cx) / (w * 0.4)) ** 2 + ((yy - cy) / (h * 0.4)) ** 2)) * nrng.uniform(5, 30)
        img += glow[..., None] * nrng.uniform(0.6, 1.0, 3)
    return img


def synth_blank(nrng, size):
    w, h = size
    kind = nrng.choice(["white", "grey", "color", "gradient"])
    if kind == "white":
        base = np.full(3, nrng.uniform(235, 255))
    elif kind == "grey":
        base = np.full(3, nrng.uniform(60, 220))
    elif kind == "color":
        base = nrng.uniform(90, 250, 3)
    else:
        base = nrng.uniform(80, 250, 3)
    img = np.ones((h, w, 3)) * base
    if kind == "gradient":
        ramp = np.linspace(0, 1, w)[None, :, None] if nrng.random() < 0.5 else np.linspace(0, 1, h)[:, None, None]
        img = img * (0.5 + 0.5 * ramp)
    return img + _noise(nrng, h, w, nrng.uniform(0.5, 4))


def synth_blur(nrng, size, src_img):
    img = src_img.convert("RGB").resize(size, Image.BILINEAR)
    radius = nrng.uniform(0.04, 0.12) * max(size)
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float64)


def synth_finger(nrng, size, bg_img):
    w, h = size
    bg = np.asarray(bg_img.convert("RGB").resize(size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.05 * max(size))), dtype=np.float64)
    tone = np.array(SKIN_TONES[nrng.integers(len(SKIN_TONES))], dtype=np.float64) * nrng.uniform(0.75, 1.1)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = nrng.uniform(0.3, 0.7) * w, nrng.uniform(0.3, 0.7) * h
    rx, ry = nrng.uniform(0.45, 0.9) * w, nrng.uniform(0.45, 0.9) * h
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # covering mask: full cover near the middle, soft edge (partial cover -> background bleeds through)
    cover = np.clip((1.15 - d) / 0.35, 0, 1) if nrng.random() < 0.7 else np.ones((h, w))
    # backlit translucent-flesh look: darker/redder core, brighter rim, plus low-frequency mottling
    core = np.clip(1 - d, 0, 1)[..., None]
    flesh = tone * (0.75 + 0.35 * (1 - core)) + core * np.array([25, -10, -12]) * nrng.uniform(0.5, 1.5)
    mott = np.asarray(Image.fromarray(nrng.uniform(0, 255, (6, 8)).astype(np.uint8)).resize((w, h), Image.BICUBIC), dtype=np.float64)[..., None]
    flesh = flesh * (0.92 + 0.16 * mott / 255.0)
    img = cover[..., None] * flesh + (1 - cover[..., None]) * bg
    return img + _noise(nrng, h, w, nrng.uniform(0.5, 4))


def make_synthetic(seed, unused_coco_names):
    nrng = np.random.default_rng(seed)
    rng = random.Random(seed)
    made = []
    with zipfile.ZipFile(RAW / "coco_val2017.zip") as z:
        pool = rng.sample(unused_coco_names, SYNTH_COUNTS["synthetic_blur"] + SYNTH_COUNTS["synthetic_finger"])
        for cat, n in SYNTH_COUNTS.items():
            for i in range(n):
                size = SYNTH_SIZES[nrng.integers(len(SYNTH_SIZES))]
                if cat == "synthetic_black":
                    arr = synth_black(nrng, size)
                elif cat == "synthetic_blank":
                    arr = synth_blank(nrng, size)
                elif cat == "synthetic_blur":
                    arr = synth_blur(nrng, size, Image.open(z.open(pool.pop())))
                else:
                    arr = synth_finger(nrng, size, Image.open(z.open(pool.pop())))
                dest = OUT / cat / f"{i:04d}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(dest, "JPEG", quality=90)
                made.append((dest, cat))
    return made


EXTRACTORS = {
    "coco": extract_coco, "places": extract_places, "dtd": extract_dtd,
    "lfw": extract_lfw, "screenshots": extract_screenshots, "documents": extract_documents,
}


def assign_splits(items, rng):
    """items: list of (path, label, source). 70/15/15 stratified by (label, source)."""
    groups = {}
    for it in items:
        groups.setdefault((it[1], it[2]), []).append(it)
    out = []
    for key in sorted(groups):
        g = sorted(groups[key])
        rng.shuffle(g)
        n = len(g)
        n_train, n_val = round(n * 0.70), round(n * 0.15)
        for i, (p, l, s) in enumerate(g):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            out.append((p, l, s, split))
    return out


def main():
    rng = random.Random(SEED)
    items = []

    # --- positives: sickle cell (all 569) -- reuse each image once. Positive/Labelled is the
    # same photos with annotation boxes drawn on, excluded exactly as in Phase 1.
    sickle = []
    for p in sorted(Path("data/raw/Positive/Unlabelled").glob("*.jpg")):
        sickle.append((str(p).replace("\\", "/"), "smear", "sickle_cell"))
    for p in sorted(Path("data/raw/Negative/Clear").glob("*.jpg")):
        sickle.append((str(p).replace("\\", "/"), "smear", "sickle_cell"))
    items += sickle

    # --- positives: malaria, stratified subsample
    per_class = MALARIA_POSITIVES // 2
    for sub in ("Parasitized", "Uninfected"):
        files = sorted(Path(f"data/malaria/raw/cell_images/{sub}").glob("*.png"))
        for p in rng.sample(files, per_class):
            items.append((str(p).replace("\\", "/"), "smear", "malaria"))

    # --- negatives
    for source, n in NEGATIVE_SOURCES.items():
        print(f"extracting {source} ({n}) ...", flush=True)
        for dest in EXTRACTORS[source](rng, n):
            items.append((str(dest).replace("\\", "/"), "not_smear", source))

    print("generating synthetic capture failures ...", flush=True)
    with zipfile.ZipFile(RAW / "coco_val2017.zip") as z:
        all_coco = sorted(x for x in z.namelist() if x.lower().endswith(".jpg"))
    unused = [n for n in all_coco if n not in USED_COCO]
    for dest, cat in make_synthetic(SEED + 1, unused):
        items.append((str(dest).replace("\\", "/"), "not_smear", cat))

    rows = assign_splits(items, rng)
    with open(ROOT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "split"])
        w.writerows(rows)

    # --- composition report
    from collections import Counter
    print("\n=== composition ===")
    print("label totals:", dict(Counter(r[1] for r in rows)))
    print("by source :", dict(Counter(r[2] for r in rows)))
    print("by split/label:", dict(Counter((r[3], r[1]) for r in rows)))


if __name__ == "__main__":
    main()
