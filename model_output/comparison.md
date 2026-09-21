# SickleScan — Model Comparison (Phase 5)

Both models share identical architecture, preprocessing, training methodology, and export format (MobileNetV2 transfer learning, frozen-head then fine-tune-top-30-layers, 224×224 bilinear resize, `(pixel-127.5)/127.5` normalization, float16 TFLite export). The only material difference is dataset size — which is exactly why the results differ the way they do.

| | Sickle Cell | Malaria |
|---|---|---|
| **Dataset size** | 569 images (422 positive / 147 negative) | 27,558 images (13,779 parasitized / 13,779 uninfected) |
| **Dataset source** | Kaggle (florencetushabe/sickle-cell-disease-dataset) | NIH/NLM LHNCBC (data.lhncbc.nlm.nih.gov), also mirrored on Kaggle |
| **Class balance** | Imbalanced (74% / 26%) — handled via class weights | Naturally balanced (50% / 50%) |
| **Test set size** | 86 images | 4,134 images |
| **Accuracy** | 94.19% | **96.30%** |
| **Sensitivity (recall, positive class)** | 93.75% | **94.34%** |
| **Specificity** | 95.45% | **98.26%** |
| **TFLite quantization** | float16 (int8 measurably worse — dropped) | float16 (int8 scored marginally higher but on far less agreement with the validated model — likely noise, not a real gain; float16 kept) |
| **Model file size** | 4.6 MB (4,792,864 bytes) | 4.6 MB (4,792,864 bytes) — identical, since it's the same architecture |

## Reading these numbers honestly

Malaria's numbers are better across the board, and the reason is straightforward: **48x more training data** (27,558 vs 569 images) and a much larger, better-balanced test set (4,134 vs 86 images) — 4,134 test images give far tighter, more trustworthy estimates than 86 do. The sickle cell numbers aren't wrong, they're just measured with much more uncertainty, and the model itself has less data to learn from. This gap is worth stating plainly in the pitch rather than letting two similar-looking percentages imply similar confidence — sensitivity of "93.75%" on 86 test images is a coarser measurement than "94.34%" on 4,134.

Thalassemia is not in this table: no lab-confirmed public dataset was found at a usable size/quality (see Phase 5 research notes) and it remains out of scope.

---

**Update (Phase 10):** the malaria model bundled in the app is now the BBBC041-trained classifier
(`model_output/malaria_bbbc041/`), not the NIH-trained model in the table above. The table above still
describes the archived NIH model accurately. The two are trained on different data and are not directly
comparable (different image type, different test sets); see `model_output/malaria_bbbc041/malaria_bbbc041_results.md`.
