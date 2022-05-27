import multiprocessing

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def compute_in_parallel(series, func, **kwargs):
    num_cores = multiprocessing.cpu_count() - 1
    results = Parallel(
        n_jobs=num_cores, backend="multiprocessing", prefer="processes"
    )(
        delayed(func)(series.iloc[i], **kwargs)
        for i in tqdm(range(series.shape[0]))
    )
    return results


def get_idx_cat_columns(X, n_modality_cat=15):
    # extract categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # extract numerical_columns
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    # extract categorical column from number
    for col in numerical_cols:

        if X[col].nunique() <= n_modality_cat:
            LOGGER.warning(
                f"numerical col = {col} interpreted as categorical, nunique = {X[col].nunique()}"
            )
            cat_cols.append(col)

    idx_cols = [X.columns.get_loc(col) for col in cat_cols]

    idx_sorted = np.argsort(idx_cols)
    cat_cols = np.array(cat_cols)[idx_sorted].tolist()
    idx_cols = np.array(idx_cols)[idx_sorted].tolist()

    idx_cols_mapping = {"cat_cols": cat_cols, "idx_cols": idx_cols}
    LOGGER.info(f"idx_cols_mapping = {idx_cols_mapping}.")
    return idx_cols_mapping
