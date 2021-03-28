import multiprocessing

from joblib import Parallel, delayed
from tqdm import tqdm


def compute_in_parallel(series, func, **kwargs):
    num_cores = multiprocessing.cpu_count() - 1
    results = Parallel(n_jobs=num_cores, backend="multiprocessing", prefer="processes")(
        delayed(func)(series.iloc[i], **kwargs) for i in tqdm(range(series.shape[0]))
    )
    return results
