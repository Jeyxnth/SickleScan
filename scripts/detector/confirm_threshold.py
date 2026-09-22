"""Phase 13: confirm the field-level operating threshold from the saved Phase 11c scores (baseline classifier, detector thr 0.3).
Prints sensitivity / negative-field flag rate at candidate thresholds and the exact in-sample threshold for >=90% sensitivity."""
import numpy as np
z = np.load("artifacts/classifier_v2/maxscores_baseline.npz")
mx, neg = z["mx"], z["is_neg"]
pos_scores = np.sort(mx[~neg])[::-1]
print(f"{int((~neg).sum())} infected fields, {int(neg.sum())} negative fields")
need = int(np.ceil(0.9 * len(pos_scores)))
t_exact = pos_scores[need - 1]
print(f"exact in-sample threshold keeping >=90% of infected fields ({need}/{len(pos_scores)}): t = {t_exact:.5f}")
for t in (0.9, 0.95, 0.98, 0.985, 0.99, float(np.round(t_exact, 4)), 0.995):
    s, f = (mx[~neg] >= t).mean(), (mx[neg] >= t).mean()
    print(f"  t={t:.4f}: infected caught {s:.1%} ({int((mx[~neg] >= t).sum())}/{int((~neg).sum())}), negative fields flagged {f:.1%} ({int((mx[neg] >= t).sum())}/{int(neg.sum())})")
