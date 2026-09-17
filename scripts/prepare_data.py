"""
Step 3-4 of Phase 1: inspect the raw dataset and build stratified
train/val/test manifests (CSV files of image path + label).

Source folders used (decided with user after inspecting the raw download):
  - data/raw/Positive/Unlabelled  -> label "positive"  (422 images, no annotation boxes)
  - data/raw/Negative/Clear       -> label "negative"  (147 images)

data/raw/Positive/Labelled is intentionally excluded: it is the exact same
422 photos as Unlabelled but with black bounding boxes drawn on the sickle
cells, which would leak the label directly into the pixels.

Output: data/splits/{train,val,test}.csv, each with columns: path,label
"""
import csv
import os
from collections import Counter

from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
SPLITS_DIR = os.path.join("data", "splits")

SOURCES = {
    "negative": os.path.join(RAW_DIR, "Negative", "Clear"),
    "positive": os.path.join(RAW_DIR, "Positive", "Unlabelled"),
}

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def collect_samples():
    samples = []
    for label, folder in SOURCES.items():
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        files = sorted(
            f for f in os.listdir(folder) if f.lower().endswith(VALID_EXT)
        )
        for fname in files:
            samples.append((os.path.join(folder, fname), label))
    return samples


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    samples = collect_samples()
    counts = Counter(label for _, label in samples)
    print("Class counts found:")
    for label, n in counts.items():
        print(f"  {label}: {n}")
    total = len(samples)
    print(f"Total images: {total}")

    paths = [p for p, _ in samples]
    labels = [l for _, l in samples]

    # 70/15/15 stratified split: first split off 70% train, then split the
    # remaining 30% evenly into val/test (15% each of the total).
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    for split_name, (sp, sl) in splits.items():
        out_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            for p, l in zip(sp, sl):
                writer.writerow([p, l])
        c = Counter(sl)
        print(
            f"{split_name}: {len(sp)} images "
            f"(positive={c.get('positive', 0)}, negative={c.get('negative', 0)}) "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
