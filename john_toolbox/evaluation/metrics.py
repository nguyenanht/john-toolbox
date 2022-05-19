from typing import Callable

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def get_optimal_threshold(
    y_true: np.array,
    y_pred: np.array,
    metric_name: str = None,
    metric_func: Callable = None,
    is_maximize: bool = True,
) -> float:
    """Calculate the optimal threshold for binary classification.

    You can optimize with the roc_auc_curve or prauc_curve by providing metric_name.
    You can also provide a custom metric function to optimize the threshold. (Please see example below)

    https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    Parameters
    ----------
    y_true : np.array
    y_pred : np.array
    metric_name : str
        Value between `roc_curve`, `prauc_curve`
    metric_func : Callable
        Function that returns a score like f1_score. the function must have argument in the following order:
        y_true, y_prob
    is_maximize : bool

    Returns
    -------
        float:
            the optimized threshold.

    Examples
    --------
    >>> from sklearn.metrics import f1_score
    >>> from john_toolbox.evaluation.metrics import get_optimal_threshold
    >>> best_threshold = get_optimal_threshold(y_true, y_pred, metric_func=f1_score)

    """

    if metric_func:
        # define thresholds
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [metric_func(y_true, to_labels(y_pred, t)) for t in thresholds]

        # get best threshold
        if is_maximize:
            ix = np.argmax(scores)
            optimize_str = "maximize"
        else:
            ix = np.argmin(scores)
            optimize_str = "minimize"

        best_thresh = thresholds[ix]
        print(
            f"Best Threshold to {optimize_str} {metric_func.__name__}={best_thresh:f}, {metric_func.__name__}={scores[ix]:.5f}"
        )
    elif metric_name == "roc_curve":
        # calculate roc curves
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        # # calculate the geometric mean that will seek a balance between sensitivy and the specificity
        # gmeans = np.sqrt(tpr * (1 - fpr))
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        print(f"Best Threshold to maximize roc_curve={best_thresh:f}")

    elif metric_name == "prauc_curve":
        # Unlike the ROC Curve, a precision-recall curve focuses on the
        # performance of a classifier on the positive (minority class) only.
        # calculate pr-curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
        print(
            f"Best Threshold to maximize prauc_curve={best_thresh:f}, F-Score={fscore[ix]:.5f}"
        )
    else:
        raise ValueError(
            "Metric name not handle. Please set value between `roc_curve`, `prauc_curve`"
        )
    return best_thresh


def to_labels(y_prob: np.array, threshold: float):
    """Apply threshold to positive probabilities to create labels.

    Parameters
    ----------
    y_prob : np.array
    threshold : float

    Returns
    -------
    np.array:
        Array of labels
    """
    return (y_prob >= threshold).astype("int")
