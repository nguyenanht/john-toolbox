import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("darkgrid")


def plot_curves(y_test, model_probs):
    # plot no skill and model precision-recall curves
    # calculate the precision-recall auc
    precision, recall, _ = precision_recall_curve(y_test, model_probs)
    fpr, tpr, _ = roc_curve(y_test, model_probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill PR AUC")
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


def plot_classification_report(y_true, y_pred, size=(10, 10), ax=None):
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
