"""metrics_utils.py - one place that computes every metric, with the positive class = label 1
(the minority class; data_pipeline maps it that way)."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score)


def compute_metrics(y_true, y_pred, scores=None, prefix=""):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    p1, r1, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if len(set(y_true)) > 1 else np.nan,
        "macro_precision": mp, "macro_recall": mr, "macro_f1": mf,
        "minority_precision": p1[1], "minority_recall": r1[1], "minority_f1": f1[1],
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    both = len(set(y_true)) > 1
    out["roc_auc"] = roc_auc_score(y_true, scores) if (scores is not None and both) else np.nan
    out["pr_auc"] = average_precision_score(y_true, scores) if (scores is not None and both) else np.nan
    return {prefix + k: v for k, v in out.items()}
