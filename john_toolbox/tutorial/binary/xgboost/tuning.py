"""
Optuna example that optimizes a classifier configuration for cancer dataset
using XGBoost.
In this example, we optimize the validation accuracy of cancer detection
using XGBoost. We optimize both the choice of booster model and its
hyperparameters.
from https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py
"""

import pprint
import optuna
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)

EVAL_METRIC = ["logloss", "auc"]


def objective_xgb(trial, X_train, X_valid, y_train, y_valid):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "n_estimators": 1000,
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1e-1, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical(
            "sample_type", ["uniform", "weighted"]
        )
        param["normalize_type"] = trial.suggest_categorical(
            "normalize_type", ["tree", "forest"]
        )
        param["rate_drop"] = trial.suggest_float(
            "rate_drop", 1e-8, 1.0, log=True
        )
        param["skip_drop"] = trial.suggest_float(
            "skip_drop", 1e-8, 1.0, log=True
        )

    xgb = XGBClassifier(**param)

    xgb.fit(
        X_train,
        y_train.to_numpy().reshape(-1),
        early_stopping_rounds=50,
        eval_set=[
            (X_train, y_train.to_numpy().reshape(-1)),
            (X_valid, y_valid.to_numpy().reshape(-1)),
        ],
        eval_metric=EVAL_METRIC,
        verbose=True,
    )

    results = xgb.evals_result()
    best_iteration = xgb.best_iteration
    print(f"Best Iteration: {best_iteration}")
    res = {
        eval_name: {key: val[xgb.best_iteration] for key, val in values.items()}
        for eval_name, values in results.items()
    }
    logger.info(res)

    # accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    auc = res["validation_1"]["auc"]

    return auc


def optimize_hyperparameter(
    X_train, X_valid, y_train, y_valid, n_trials=50, timeout=None
):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        func=lambda trial: objective_xgb(
            trial=trial,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    logger.info(f"Number of finished trials: {len(study.trials)}")
    trial = study.best_trial

    logger.info(f"Best trial value: {trial.value}")
    pprint.pprint(trial.params)
    return trial
