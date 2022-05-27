import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

sns.set_style("darkgrid")


def plot_auc_curves(y_test, model_probs):
    """Plot no skill, model precision-recall curves and roc auc curves

    Parameters
    ----------
    y_test :
    model_probs :

    Returns
    -------

    """

    # calculate the precision-recall auc
    precision, recall, _ = precision_recall_curve(y_test, model_probs)
    fpr, tpr, _ = roc_curve(y_test, model_probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    # plot the no skill precision-recall curve
    plt.plot(
        [0, 1], [no_skill, no_skill], linestyle="--", label="No Skill PR AUC"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill ROC AUC")
    # plot model precision-recall curve
    plt.plot(recall, precision, marker=".", label=f"PR AUC ({pr_auc})")
    plt.plot(fpr, tpr, marker=".", label=f"ROC AUC ({roc_auc})")
    # axis labels
    plt.xlabel("Recall/TPR")
    plt.ylabel("Precision/FPR")
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def plot_classification_report(y_true, y_pred, size=(5, 5), ax=None):
    """
    https://stackoverflow.com/a/44188254
    """
    plt.figure(figsize=size)

    xticks = ["precision", "recall", "f1-score", "support"]
    yticks = list(np.unique(y_true))
    yticks += ["avg"]

    rep = np.array(precision_recall_fscore_support(y_true, y_pred)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(
        rep,
        annot=True,
        cbar=False,
        xticklabels=xticks,
        yticklabels=yticks,
        ax=ax,
        fmt="g",
    ).set_title(f"Accuracy score : {round(accuracy_score(y_true, y_pred), 4)}")


def compare_eval_result_xgb(
    model,
    eval_names: list,
    eval_metrics: list,
    is_custom_eval_metric: bool = False,
    size: tuple = (5, 5),
):
    results = model.evals_result()

    for metric in eval_metrics:
        n_epochs = None
        fig, ax = plt.subplots(figsize=size)

        for i, eval_name in enumerate(eval_names):
            x = results[f"validation_{i}"][metric]
            n_epochs = len(x)
            x_axis = range(0, n_epochs)
            ax.plot(x_axis, x, label=eval_name)
        ax.legend()
        plt.text(
            1,
            0,
            f"n_epochs: {n_epochs}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=dict(facecolor="red", alpha=0.5),
        )
        plt.ylabel(f"{metric}")
        plt.title(f"{type(model).__name__}_{metric}")
        plt.show()


def plot_cm(y_true, y_pred, figsize=(5, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt="", ax=ax)
