# Phase 12 — borrowed pretrained malaria detector: source / license check (Task 1) and why Task 2 was not run

**Status: stopped after Task 1. No model was run, nothing was downloaded, no API key was used, and Photos 1-4 were not touched.** The BBBC041 validation in Task 2 cannot produce a valid result for the leading candidate (details below), so I am asking rather than running a contaminated or credential-dependent test.

## Task 1.1 — universe.roboflow.com/mmpk/malaria-detection-4dn5i
| Question | Finding (from the public page; JSON-LD metadata; no login) |
|---|---|
| What it is | "Malaria Detection" by uploader **Mmpk**, 1,328 images, 12 dataset versions, 3 trained models, last updated 2023 |
| Source data | **BBBC041 re-upload, in effect.** 1,328 images (the exact BBBC041 annotated-image count) and the exact BBBC041 seven classes (ring, difficult, gametocyte, leukocyte, red blood cell, schizont, trophozoite). The page names no source and gives no citation; I have not compared pixels because the dataset download needs an account |
| Stated license | "Public Domain" (metadata: CC0 1.0). **This is not reliable:** BBBC041 itself is **CC BY-NC-SA 3.0** (Attribution, NonCommercial, ShareAlike, by Jane Hung, recommended citation Ljosa et al. 2012). A re-uploader cannot relicense data they do not own, so the honest terms are BBBC041's, not CC0 |
| The model | "Roboflow 2.0 Object Detection (Fast)", trained on dataset version 5 (**1,328 images = all of them**), checkpoint from Mar 2023. **No mAP / precision / recall shown on the page** |
| How it is served | Hosted API only (`serverless.roboflow.com`, needs an API key). Listed deployment targets: Raspberry Pi, NVIDIA Jetson, Docker, web page, iOS, Python SDK. **No TFLite/Android export is listed for the model.** Weight download is not offered on the public page (whether it is possible with an account is unverified) |
| Dataset export formats | YOLO (v3), COCO, Pascal VOC, TFRecord, Keras, Darknet, KITTI, CreateML, etc. **TFLite is not among the listed formats** (so a conversion step would be needed, and only if weights can be obtained at all). Dataset download is behind a Roboflow account (`requiresSubscription: true` in the page metadata) |

## Task 1.2 — other candidates
| Candidate | Source | License shown | Verdict |
|---|---|---|---|
| **rifqi-riset/malaria-detection-xbfdy** | 1,328 images (BBBC041 again), only 4 classes (ring/gametocyte/schizont/trophozoite; **no RBC class**), Roboflow 2.0 Fast checkpoint "COCOv6n", 21 versions | Public Domain (same unreliable relabel) | Same problems as above: hosted, trained on all 1,328 images, no metrics shown |
| rifqi-riset/malaria-detection-stcqo | 1,438 images, classes named only "0-5", 3,708 images in the trained version, source unstated | CC BY 4.0 | Provenance unknown; cannot be attributed properly |
| biomedical-0nxme/malaria-dataset-bjdff | 500 images, classes infected/uninfected (cell-level, not field images), shows mAP@50 98.4% / P 98.9% / R 98.9%, trained on 1,200 images | CC BY 4.0 | Different data type (single-cell style), source unstated; the high mAP is on its own split, not comparable |
| rocha-zxgmt/malaria-gl6ep, scunetdatatestsets/malaria-dataset-kffy4 | 1,182 images, one class "object", no trained model | (none) / CC BY 4.0 | Datasets only, and unclear provenance (1,182 matches a thick-smear set, not BBBC041) |
| biomedicallab/malaria-dataset-rqrpe | page could not be retrieved (bot challenge) | unknown | not evaluated |
| **YOLOv4 thin-smear paper** (Sukumarran et al., *Parasites & Vectors* 2024;17:188, doi:10.1186/s13071-024-06215-7, CC BY 4.0) | Trained on **MP-IDB** (210 thin-smear images, several Plasmodium species) and a **private** UNIMAS Sarawak set (472 images). **BBBC041 is not used.** | Paper CC BY 4.0 | **No code or weights are released** (Zenodo holds only the article PDF and table files; the data-availability statement points to the public MP-IDB dataset and says the second set is available from the Centre). Also reports **precision 59% on its own test data (486 false positives on 472 images, one to two per image, mostly faint stains and small debris)**: the same false-alarm problem we measured |

## Why Task 2 was not run
1. **It would not be a valid held-out test.** The hosted models were trained on all 1,328 BBBC041 images. Our 133 infected + 30 negative validation fields are BBBC041 images, so they are at best a mix of the model's training and its own unknown validation/test images. A good score would tell us the model memorised its data; a bad score would be informative but only as an upper bound. Neither answers "is it better than our pipeline on unseen fields".
2. **It needs an API key and sends images to a third-party server.** I do not handle credentials on my own, and the request would upload BBBC041 images (CC BY-NC-SA) to Roboflow. I would want your explicit go-ahead and key handling instructions.
3. **There is no on-device path.** Hosted API, no TFLite export listed. Even a perfect score would not give something we can ship in the offline Android app without a weights download I cannot confirm exists, plus conversion and fidelity work.

## Side finding that affects our own app (needs your attention)
BBBC041 is **CC BY-NC-SA 3.0**. Our shipped malaria classifier (Phase 10) and the Phase 11 detector are trained on it. At minimum the README and references slide must carry the recommended citation ("We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]"), credit Jane Hung, and Hung et al. 2018 (arXiv:1804.09548). The NonCommercial and ShareAlike terms need a decision from you (an ideathon prototype is plausibly non-commercial; whether ShareAlike extends to trained weights is a legal question I have not resolved). This was not previously verified; my earlier notes said only that redistribution terms were unverified.

## Proposed options
A. **Drop this line.** None of the candidates offers a licensed, downloadable, TFLite-exportable, independently testable model. Keep our own pipeline; it already has measured numbers (Phases 11a-c).
B. **Evaluate one hosted model as a reference only** (mmpk, hosted API) on the 163 validation fields, clearly labelled *contaminated, upper-bound only*, not a basis for replacing our pipeline. Needs your API key and consent to upload the images.
C. **Test against a genuinely different labelled set** (e.g. MP-IDB, P. falciparum etc., public on GitHub): not an equivalent comparison (different species/stain), but the only way to get an uncontaminated number for a BBBC041-trained model. I have not checked its license or size.

Sources: Roboflow Universe pages listed above; BBBC041 page (bbbc.broadinstitute.org/BBBC041); Hung et al. 2018, arXiv:1804.09548; Sukumarran et al. 2024 via Zenodo record 11074830.
