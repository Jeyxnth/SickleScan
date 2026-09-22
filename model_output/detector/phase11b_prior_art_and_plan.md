# Phase 11b — prior art on joint detection+classification, compatibility, and proposed plan (nothing built)

## Task 1 — what the BBBC041 authors actually did (Hung et al., arXiv 1804.09548)

**Classes.** Seven labels: RBC, leukocyte, gametocyte, ring, trophozoite, schizont, and "difficult" (cells not clearly in a class; *ignored in training*). RBCs are ~97% of ~100,000 cells. Training/validation data come from Manaus + Thailand labs; a **Brazil lab set is held out as the test set** (i.e. the same shifted-domain split we found in `test.json`).

**Models and what they report.**
| Model | Result reported |
|---|---|
| Baseline: CellProfiler segmentation + random forest | 50% accuracy (excluding background, RBC, difficult) |
| **One-stage Faster R-CNN (joint detect + classify all classes)**, AlexNet backbone, score threshold 0.65 | **59% accuracy** (same exclusions); confused between infected stages; blamed on class imbalance |
| **Two-stage: Faster R-CNN finds objects as "RBC vs other", then a separate AlexNet classifier on the "other" crops** | 98% accuracy (same exclusions) — this is their best model |
| Two human experts vs each other | 72% accuracy on non-difficult infected cells; per-class F1 44-92% |

**Their best model is detect-then-classify-separately — the same family we already built — not joint training.** Their joint one-stage model was the *worse* one.

**Do they report a false-positive rate on clean negative fields? No.** No precision, recall, mAP, or negative-field evaluation appears anywhere. Their accuracy figures explicitly exclude RBCs and background, and the confusion matrix (their Fig 7a) has no RBC class, so RBC-to-infected errors are invisible in it. The only false-positive evidence is one count table for the joint model on the test set (their Fig 6a, threshold 0.65), which is **net counts, not matched detections**:

| Class | Model count | Ground truth |
|---|---|---|
| RBC | 19,112 | 19,604 |
| trophozoite / schizont / ring / gametocyte | 664 / 39 / 227 / 74 = **1,004** | 561 / 28 / 88 / 75 = **752** |
| leukocyte | 49 | 28 |
So the joint model over-predicts infected cells by at least 252 (33%) on ~19,600 cells, with rings 2.6x over. That is a lower bound of ~1.3% of cells wrongly labelled infected (net of any misses) — the same order as our 1.6-1.9% per cell. It is weak evidence (counts only), but it points to **the same wall, not a way around it.** They also did not test another lab beyond the one test set and list cross-lab robustness as future work; human annotators themselves disagree substantially, which suggests label noise sets a ceiling for everyone using this dataset.

**Working code.**
| Repo | Finding |
|---|---|
| broadinstitute/keras-rcnn | Pins TensorFlow 1.13.1, Keras 2.2.4, numpy 1.16.2; last push May 2020; README example is a toy shapes dataset, no BBBC041 pipeline or results |
| sriluk9/MalariaCells-ObjectDetection-Using-FasterRCNN | Jupyter notebooks, `keras-frcnn`-style VGG Faster R-CNN, TF1 on Python 3.6 ("Using TensorFlow backend"); its evaluation prints per-image AP with **`mAP = nan`** and no aggregate or false-positive numbers; last push Jan 2020 |
| ErickDiaz/bioinformatic_thesis_project | U-Net segmentation plus a YOLO demo video; no versions, no metrics in the README; last push Feb 2020 |
None gives a reproducible metric we could compare against.

## Task 2 — compatibility with this machine (Python 3.10, TF 2.16 / Keras 3, CPU, Windows)
- **keras-rcnn and both reproductions: not usable.** TF 1.x / Keras 2.2 / Python 3.6-3.7 era; adopting them means a second incompatible environment, the same class of blocker that ruled out TFLite Model Maker. Their output would also need a TFLite-conversion path that TF1 Faster R-CNN (dynamic region-proposal ops) does not have cleanly.
- **Modern packages (checked with `pip install --dry-run`, nothing installed):** `keras-cv 0.9.0` and `keras-hub 0.25.1` both resolve without touching TF/Keras/numpy. But I have *not* verified that either offers a detector that trains acceptably on CPU and exports to TFLite, and keras-cv is a stale package. Not worth a third pivot.
- **PyTorch/torchvision Faster R-CNN:** would work in principle but is a large new dependency and Android would need an ONNX/TF-conversion chain — a new failure surface. Not recommended.
- **Conclusion:** adopt neither Faster R-CNN nor keras-rcnn. The closest viable modern equivalent is extending our own working CenterNet-style MobileNetV2 detector.

## What the evidence says about the hypothesis (field context lost in isolated crops)
- Joint training is not shown to avoid the false-positive wall: the authors' joint model was worse than their two-stage one, and its counts over-predict infection ~33%.
- Our own numbers: the extra false positives come from the classifier's ~1.8% per-cell rate; 26 of the 41 flagged cells sit on cells labelled ordinary RBC and 15 on cells with no label. **I have not looked at them.** If some are real unannotated parasites, part of the "false-positive rate" is label noise that no architecture can fix, and the true target rate is lower than measured.
- **Field context is a live hypothesis, not a proven one.** A joint head sees a much larger receptive field and every RBC in every training field (~97% of ~80,000 boxes), instead of 6,000 subsampled RBC crops — plausible reasons it could help. There is also a practical benefit independent of accuracy: today one photo costs 1 detector pass + ~73 classifier passes (2,184 cells / 30 negative fields); a joint model is one pass and one bundled model.

## Task 3 — proposed next step (no training code written)
**Recommendation: (a) a modern joint version of what we have, gated by two cheap experiments that tell us whether it is worth building. Not (b).**

**Step 0 — look at the data (minutes, no training).** Montage of the 41 flagged cells from the negative validation fields. Outcome sets the honest ceiling: if many are real unlabelled parasites, the target changes and the "problem" is partly the labels.

**Step 1 — two cheap tests of the context hypothesis, on the existing classifier setup (est. 10-20 min each; I have not timed this trainer):**
- 1a. Retrain the crop classifier with **hard negatives** mined from the detector's own false positives on the *training* split (the standard fix for a per-cell false-positive rate).
- 1b. Retrain the classifier on **wider crops** (padding ~2x instead of 1.15x) so it sees neighbouring cells. If wider context lowers false positives at equal sensitivity, that supports the joint approach; if not, joint training is unlikely to be the fix.

**Step 2 (only if step 1 says context helps, or you want the single-model on-device benefit) — build the joint model:** add one **"infected" heatmap channel** to the existing CenterNet head (same MobileNetV2 + FPN, same data pipeline; binary infected vs uninfected, not the 4-stage classification that hurt Hung et al.), trained with focal loss on all cells in full-field crops. One forward pass gives cell locations and an infection score per cell. Est. ~1 h CPU training (the detector took 58 min). Includes TFLite-conversion check as an explicit gate.

**Success criteria — fixed now, before any training, so we can't move the goalposts** (evaluated on the same held-out BBBC041 validation split used in the last test, so results are comparable with the current pipeline):
- Sensitivity on the 133 infected validation fields >= 90% with **negative-field flag rate <= 15%** (today: 92-94% at 30%).
- Per-cell false-positive rate reported with 95% intervals; no threshold chosen on the negative fields themselves.
- Held-out negatives are few (30 fields, 48 if "difficult" is allowed; 3 in the test split), so intervals will be wide (~±17 points). **Decision for you: keep the current split (comparable, wide intervals) or re-split with many more held-out negative fields (tighter intervals, but the current two-stage baseline must then be retrained for a fair comparison).** I recommend keeping the split.

**Known limits none of these options addresses:** score compression under a different microscope (test-split sensitivity 41% at 0.5 for the classifier), P. vivax only, and thin-smear-only photos (Photos 1, 3, 4). If the flag-rate target is missed after step 2, the remaining lever is labelled negative fields from *your* photo setup, not more architecture.

Sources: Hung et al. 2018, arXiv:1804.09548 (text and Figs 5-8 read from the PDF); GitHub repository metadata and notebooks for the three repos above.
