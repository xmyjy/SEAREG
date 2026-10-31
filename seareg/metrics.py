"""Ranking evaluation for held-out spot–peak observations."""

from __future__ import annotations

import numpy as np


def _labels(pos, neg):
    import torch
    def array(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        a = np.asarray(x, dtype=np.float64)
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError("scores must be finite 1D arrays")
        return a

    positive, negative = array(pos), array(neg)
    return (
        np.concatenate([np.ones(len(positive)), np.zeros(len(negative))]),
        np.concatenate([positive, negative]),
        len(positive),
        len(negative),
    )


def best_f1_threshold(pos, neg):
    from sklearn import metrics
    y, s, npos, nneg = _labels(pos, neg)
    if not npos or not nneg:
        raise ValueError("threshold selection requires both score populations")
    p, r, t = metrics.precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


def binary_metrics(pos, neg, threshold=None, threshold_source="not_selected"):
    from sklearn import metrics
    y, s, npos, nneg = _labels(pos, neg)
    result = {
        "status": "EVALUATED" if npos and nneg else "NOT_EVALUATED",
        "n_positive": npos,
        "n_negative": nneg,
        "roc_auc": None,
        "pr_auc": None,
        "average_precision": None,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "precision": None,
        "recall": None,
        "accuracy": None,
        "f1": None,
        "score_protocol": "raw_logits_v1",
    }
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not npos or not nneg:
        return result
    p, r, _ = metrics.precision_recall_curve(y, s)
    result.update(
        roc_auc=float(metrics.roc_auc_score(y, s)),
        pr_auc=float(metrics.auc(r, p)),
        average_precision=float(metrics.average_precision_score(y, s)),
    )
    if threshold is not None:
        prediction = s >= threshold
        result.update(
            precision=float(metrics.precision_score(y, prediction, zero_division=0)),
            recall=float(metrics.recall_score(y, prediction)),
            accuracy=float(metrics.accuracy_score(y, prediction)),
            f1=float(metrics.f1_score(y, prediction, zero_division=0)),
        )
    return result


def recall_at_k(spots, peaks, test_positive, train_positive, ks=(20, 50, 100), candidates="unseen"):
    import torch
    with torch.no_grad():
        if candidates not in ("unseen", "all"):
            raise ValueError("unknown candidate protocol")
        if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in ks):
            raise ValueError("K must be a positive integer")
        spots = torch.as_tensor(spots)
        peaks = torch.as_tensor(peaks)
        if (
            spots.ndim != 2
            or peaks.ndim != 2
            or spots.shape[1] != peaks.shape[1]
            or not torch.isfinite(spots).all()
            or not torch.isfinite(peaks).all()
        ):
            raise ValueError("invalid recall embedding matrices")

        def edges(value):
            a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if a.ndim != 2 or a.shape[0] != 2:
                raise ValueError("positive edges must be 2xE local indices")
            if a.size and not np.issubdtype(a.dtype, np.integer):
                raise ValueError("recall edge indices must have integer dtype")
            if a.size and (a.min() < 0 or a[0].max() >= len(spots) or a[1].max() >= len(peaks)):
                raise ValueError("recall edge outside node universe")
            return a.astype(np.int64, copy=False)

        test, train = edges(test_positive), edges(train_positive)
        truth = {}
        known = {}
        for u, p in test.T:
            truth.setdefault(int(u), set()).add(int(p))
        for u, p in train.T:
            known.setdefault(int(u), set()).add(int(p))
        result = {k: [] for k in ks}
        for u, positive in truth.items():
            scores = spots[u] @ peaks.T
            eligible = len(peaks)
            if candidates == "unseen":
                blocked = known.get(u, set())
                if positive & blocked:
                    raise ValueError("test positives overlap training candidates")
                if blocked:
                    scores[list(blocked)] = -torch.inf
                eligible -= len(blocked)
            if eligible == 0:
                raise ValueError("no eligible recall candidates")
            for k in ks:
                top = set(torch.topk(scores, min(k, eligible)).indices.cpu().tolist())
                result[k].append(len(top & positive) / len(positive))
        return {k: float(np.mean(values)) if values else None for k, values in result.items()}
