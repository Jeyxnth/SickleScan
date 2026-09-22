"""Paired bootstrap over fields + matched-operating-point comparison from the saved per-field max cell scores.
Usage: python scripts/classifier_v2/bootstrap_compare.py"""
import os
import numpy as np

D = os.path.join("artifacts", "classifier_v2")
names = ["baseline", "wide", "hardneg"]
Z = {n: np.load(os.path.join(D, f"maxscores_{n}.npz")) for n in names}
neg = Z["baseline"]["is_neg"]
S = {n: Z[n]["mx"] for n in names}


def auc(m, isn):
    pos, ng = m[~isn], m[isn]
    return ((pos[:, None] > ng[None, :]).sum() + 0.5 * (pos[:, None] == ng[None, :]).sum()) / (len(pos) * len(ng))


rng = np.random.default_rng(0)
pi, ni = np.where(~neg)[0], np.where(neg)[0]
B = 2000
boot = {n: [] for n in names}
for _ in range(B):
    idx = np.concatenate([rng.choice(pi, len(pi)), rng.choice(ni, len(ni))])
    for n in names:
        boot[n].append(auc(S[n][idx], neg[idx]))
print("Image-level AUC (max cell score), 95% bootstrap over fields:")
for n in names:
    b = np.array(boot[n]); print(f"  {n:9s} {auc(S[n], neg):.3f}  [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
for a, b_ in (("hardneg", "baseline"), ("wide", "baseline")):
    d = np.array(boot[a]) - np.array(boot[b_])
    print(f"  AUC diff {a} - {b_}: {np.mean(d):+.3f}  [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

print("\nMatched operating points (same sensitivity, compare negative-field flag rate):")
h = S["hardneg"]
sens_h = (h[~neg] >= 0.5).mean(); fl_h = (h[neg] >= 0.5).mean()
print(f"  hardneg @0.5: sens {sens_h:.1%}, neg fields flagged {fl_h:.0%} ({int(round(fl_h * neg.sum()))}/{neg.sum()})")
b = S["baseline"]
for t in (0.5, 0.9, 0.95, 0.98, 0.985, 0.99, 0.995, 0.999):
    print(f"  baseline @{t}: sens {(b[~neg] >= t).mean():.1%}, neg fields flagged {(b[neg] >= t).mean():.0%} ({int((b[neg] >= t).sum())}/{neg.sum()})")
