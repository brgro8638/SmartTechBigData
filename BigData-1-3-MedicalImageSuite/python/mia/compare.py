"""Agreement metrics between two label maps (implementations or ground truth)."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def contingency(a, b):
    a, b = np.ravel(a), np.ravel(b)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def match_labels(reference, other):
    """Map each label of `other` to the reference label it overlaps most
    (Hungarian algorithm on the overlap table). Label numbers from different
    implementations or from ground truth are arbitrary, so they must be
    matched before pixels can be compared."""
    table = contingency(reference, other)
    rows, cols = linear_sum_assignment(-table)
    mapping = np.full(table.shape[1], -1, dtype=np.int64)
    mapping[cols] = rows
    return mapping[np.asarray(other)]


def agreement(reference, other):
    reference, other = np.asarray(reference), np.asarray(other)
    mapped = match_labels(reference, other)
    differ = mapped != reference
    return {
        "matched agreement": float(1 - differ.mean()),
        "ARI": float(adjusted_rand_score(reference.ravel(), other.ravel())),
        "differing pixels": int(differ.sum()),
        "difference_mask": differ,
    }


def label_stats(labels):
    return {"labels": int(labels.max()) + 1, "foreground %": float((labels > 0).mean() * 100)}
