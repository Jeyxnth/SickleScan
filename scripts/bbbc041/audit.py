"""BBBC041 data audit: annotation format, image sizes, class distribution, box sizes."""
import collections
import json
import os

import numpy as np

ROOT = os.path.join("data", "bbbc041", "raw", "malaria")


def load(split):
    return json.load(open(os.path.join(ROOT, f"{split}.json")))


if __name__ == "__main__":
    for split in ("training", "test"):
        d = load(split)
        print(f"\n===== {split}.json: {len(d)} images =====")
        if split == "training":
            print("first record keys:", list(d[0].keys()), "| image keys:", list(d[0]["image"].keys()))
            print("example:", json.dumps(d[0])[:400])
        shapes = collections.Counter((r["image"]["shape"]["r"], r["image"]["shape"]["c"]) for r in d)
        print("image shapes (rows x cols):", dict(shapes))
        cats = collections.Counter(o["category"] for r in d for o in r["objects"])
        print("class counts:", dict(cats), "| total boxes:", sum(cats.values()))
        n_obj = np.array([len(r["objects"]) for r in d])
        print(f"boxes/image: mean {n_obj.mean():.1f} median {np.median(n_obj):.0f} min {n_obj.min()} max {n_obj.max()}; images with 0 boxes: {(n_obj == 0).sum()}")
        per_cat = collections.defaultdict(list)
        for r in d:
            for o in r["objects"]:
                b = o["bounding_box"]
                h = b["maximum"]["r"] - b["minimum"]["r"]
                w = b["maximum"]["c"] - b["minimum"]["c"]
                per_cat[o["category"]].append((w, h))
        for c, v in per_cat.items():
            v = np.array(v)
            print(f"  {c:16s} n={len(v):6d} box w median {np.median(v[:, 0]):.0f}px (p10 {np.percentile(v[:, 0], 10):.0f}, p90 {np.percentile(v[:, 0], 90):.0f}) h median {np.median(v[:, 1]):.0f}px")
        # how many images contain each infected stage
        stage = collections.Counter()
        for r in d:
            for c in {o["category"] for o in r["objects"]}:
                stage[c] += 1
        print("images containing class:", dict(stage))
    imgs = os.listdir(os.path.join(ROOT, "images"))
    print("\nfiles in images/:", len(imgs), collections.Counter(f.rsplit('.', 1)[-1] for f in imgs))
