"""Metrics. Attack = positive class.

Reported for every (model, source, target):
  F1, precision, detection rate (recall on attacks), false-positive rate
  (share of benign flows flagged), balanced accuracy, MCC, ROC-AUC, PR-AUC,
  and per-attack-class detection rate.

Threshold: chosen on the SOURCE validation set (max F1) and then frozen - this is
what a deployed detector would do. We also report an "oracle" F1 using the best
threshold on the target test set; it is NOT achievable in practice, but the gap
between the two tells us whether the model's ranking still works on the new
network and only its threshold is off, or whether the ranking itself breaks.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)


def best_f1_threshold(y, score):
    if y.min() == y.max():
        return 0.5, float("nan")
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5, float(f1[i])


def binary_metrics(y, score, thr):
    y = np.asarray(y).astype(int)
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    dr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * dr / (prec + dr) if (prec + dr) and not np.isnan(dr) else 0.0
    both = y.min() != y.max()
    return {
        "f1": f1, "precision": prec, "detection_rate": dr, "fpr": fpr,
        "balanced_acc": (dr + (1 - fpr)) / 2 if both else float("nan"),
        "mcc": matthews_corrcoef(y, pred) if both else float("nan"),
        "roc_auc": roc_auc_score(y, score) if both else float("nan"),
        "pr_auc": average_precision_score(y, score) if both else float("nan"),
    }


def per_class_rates(attack_labels: pd.Series, score, thr) -> pd.DataFrame:
    """Share of flows flagged as attack, per class. For 'Benign' this is the
    false-alarm rate; for every attack class it is that class's detection rate."""
    flagged = pd.Series((score >= thr).astype(float), index=attack_labels.index)
    g = flagged.groupby(attack_labels.values)
    return pd.DataFrame({"n": g.size(), "flagged_rate": g.mean()}).rename_axis("attack_class").reset_index()
