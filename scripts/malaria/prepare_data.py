"""
Inspect the raw malaria dataset and build stratified train/val/test
manifests, matching Phase 1's methodology exactly (70/15/15 stratified
split, same random_state).

Source: NIH/NLM LHNCBC Malaria Cell Images dataset, downloaded directly
from https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
(the same data also mirrored on Kaggle as
iarunava/cell-images-for-detecting-malaria -- both confirmed live before
downloading).

  - data/malaria/raw/cell_images/Parasitized -> label "parasitized" (13,780 images)
  - data/malaria/raw/cell_images/Uninfected   -> label "uninfected"  (13,778 images)

No labelled/unlabelled duplication issue here (unlike the sickle cell
dataset) -- this is already a clean, pre-balanced two-folder split.

Output: data/malaria/splits/{train,val,test}.csv, columns: path,label
"""
import csv
import os
from collections import Counter

from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "malaria", "raw", "cell_images")
SPLITS_DIR = os.path.join("data", "malaria", "splits")

SOURCES = {
    "uninfected": os.path.join(RAW_DIR, "Uninfected"),
    "parasitized": os.path.join(RAW_DIR, "Parasitized"),
}

VALID_EXT = (".png",)


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
            f"(parasitized={c.get('parasitized', 0)}, uninfected={c.get('uninfected', 0)}) "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
