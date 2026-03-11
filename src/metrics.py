from dataclasses import dataclass
from typing import Iterable, List, Mapping, Set


@dataclass
class PRMetrics:
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def precision_recall_for_video(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> PRMetrics:
    """Compute precision/recall for a single multi-label video."""
    true_set: Set[str] = {a.lower() for a in y_true}
    pred_set: Set[str] = {a.lower() for a in y_pred}

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return PRMetrics(precision=precision, recall=recall, tp=tp, fp=fp, fn=fn)


def aggregate_micro_metrics(per_video: List[PRMetrics]) -> PRMetrics:
    """Aggregate micro-averaged precision/recall over many videos."""
    tp = sum(m.tp for m in per_video)
    fp = sum(m.fp for m in per_video)
    fn = sum(m.fn for m in per_video)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return PRMetrics(precision=precision, recall=recall, tp=tp, fp=fp, fn=fn)


def labels_from_row(row: Mapping[str, int]) -> List[str]:
    """Convert a CSV row with 0/1 columns into a list of present labels.

    Assumes label columns are already filtered to the action set.
    """
    return [k for k, v in row.items() if int(v) == 1]

