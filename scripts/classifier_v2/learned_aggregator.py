"""
Phase 11c: does a small learned field-level aggregator beat hand-picked decision rules?
Uses the saved per-cell classifier scores (cells_<name>.npz: 133 infected + 30 negative BBBC041 val fields, detector score >= 0.3).
Field features: max score, mean of top-3, mean of top-5, count >= 0.5 / 0.7 / 0.9, number of cells.
Models: logistic regression (L2, class-balanced), regularisation chosen by an INNER 3-fold CV (nested), and a 2-feature version.
Protocol (no tuning on held-out fields): repeated stratified 5-fold CV (20 repeats). For every method (learned or hand rule)
the operating threshold is set on the TRAINING folds only (largest threshold reaching the target sensitivity there) and applied
to the held-out fold; the pooled held-out predictions give the REALISED sensitivity and negative-field flag rate.
Usage: python scripts/classifier_v2/learned_aggregator.py [baseline|hardneg]
"""
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
NAME = sys.argv[1] if len(sys.argv) > 1 else "baseline"
THR_DET = 0.3
Z = np.load(os.path.join("artifacts", "classifier_v2", f"cells_{NAME}.npz"), allow_pickle=True)
y = (~Z["is_neg"]).astype(int)          # 1 = infected field
cell_scores = [p[sc >= THR_DET] for p, sc in zip(Z["p"], Z["sc"])]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def features(q):
    s = np.sort(q)[::-1] if len(q) else np.zeros(1)
    top = lambda k: s[:k].mean()
    return [logit(s[0]), logit(top(3)), logit(top(5)), np.log1p((q >= .5).sum()), np.log1p((q >= .7).sum()),
            np.log1p((q >= .9).sum()), np.log1p(len(q))]


F = np.array([features(q) for q in cell_scores])
kth = lambda q, k: (np.sort(q)[::-1][k - 1] if len(q) >= k else 0.0)
rule_scores = {"hand: any cell >= t (max)": np.array([kth(q, 1) for q in cell_scores]),
               "hand: >=2 cells >= t (2nd highest)": np.array([kth(q, 2) for q in cell_scores]),
               "hand: >=3 cells >= t (3rd highest)": np.array([kth(q, 3) for q in cell_scores])}
print(f"classifier variant: {NAME} | fields: {int(y.sum())} infected, {int((1 - y).sum())} negative | detector thr {THR_DET}")


def auc(s, yy):
    pos, neg = s[yy == 1], s[yy == 0]
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def make_lr(cols):
    return lambda: make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, scoring="roc_auc",
                                                                       class_weight="balanced", max_iter=2000))


learned = {"learned LR, 7 features": (make_lr(None), list(range(7))),
           "learned LR, 2 features (max, n cells)": (make_lr(None), [0, 6])}
methods = list(rule_scores) + list(learned)
TARGETS = (0.85, 0.90, 0.95)
R = 20
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=0)
oof = {m: np.zeros((R, len(y))) for m in methods}
realised = {m: {t: {"tp": np.zeros(R), "fp": np.zeros(R)} for t in TARGETS} for m in methods}
pred = {m: {t: np.zeros((R, len(y)), bool) for t in TARGETS} for m in methods}
for i, (tr, te) in enumerate(rskf.split(F, y)):
    r = i // 5
    for m in methods:
        if m in rule_scores:
            s_all = rule_scores[m]; s_tr, s_te = s_all[tr], s_all[te]
        else:
            mk, cols = learned[m]
            model = mk().fit(F[tr][:, cols], y[tr])
            s_tr, s_te = model.decision_function(F[tr][:, cols]), model.decision_function(F[te][:, cols])
        oof[m][r, te] = s_te
        for t in TARGETS:
            thr = np.quantile(s_tr[y[tr] == 1], 1 - t, method="lower")  # largest thr keeping >= t of TRAIN positives
            pred[m][t][r, te] = s_te >= thr
print("\n== Held-out AUC over the 20 CV repeats (pooled out-of-fold scores) ==")
print(f"  parameter-free reference: max cell score, all fields: {auc(rule_scores['hand: any cell >= t (max)'], y):.3f}")
for m in methods:
    a = np.array([auc(oof[m][r], y) for r in range(R)])
    print(f"  {m:42s} {a.mean():.3f}  (repeat-to-repeat sd {a.std():.3f}, range {a.min():.3f}-{a.max():.3f})")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


print("\n== REALISED operating points (threshold set on training folds only; mean over 20 repeats) ==")
n_neg = int((1 - y).sum()); n_pos = int(y.sum())
for t in TARGETS:
    print(f"\n-- target sensitivity {t:.0%} on training folds --")
    for m in methods:
        sens = np.array([pred[m][t][r][y == 1].mean() for r in range(R)])
        fl = np.array([pred[m][t][r][y == 0].mean() for r in range(R)])
        lo, hi = wilson(fl.mean() * n_neg, n_neg)
        print(f"  {m:42s} infected caught {sens.mean():5.1%} (sd {sens.std():.1%}) | negative fields flagged {fl.mean():5.1%} "
              f"[Wilson 95% {lo:.0%}-{hi:.0%}] (sd across splits {fl.std():.1%})")
print("\n== Fixed hand-picked rules, no fitting (all fields) ==")
for name, k, thr in (("any cell >= 0.5", 1, .5), ("any cell >= 0.9", 1, .9), (">=2 cells >= 0.5", 2, .5), (">=3 cells >= 0.5", 3, .5)):
    s = np.array([kth(q, k) for q in cell_scores]) >= thr
    print(f"  {name:20s} infected caught {s[y == 1].mean():5.1%} | negative fields flagged {s[y == 0].mean():5.1%} ({int(s[y == 0].sum())}/{n_neg})")

# fit on all data once just to show what the model relies on (not used for any performance claim)
mk, cols = learned["learned LR, 7 features"]
full = mk().fit(F[:, cols], y)
coef = full[-1].coef_[0]
names = ["logit(max)", "logit(mean top3)", "logit(mean top5)", "log1p(n>=0.5)", "log1p(n>=0.7)", "log1p(n>=0.9)", "log1p(n cells)"]
print("\nCoefficients of the model fitted on ALL fields (standardised features; descriptive only, C =", full[-1].C_[0], "):")
for n_, c in zip(names, coef):
    print(f"  {n_:18s} {c:+.2f}")
