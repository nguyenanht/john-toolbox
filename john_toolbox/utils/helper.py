import sys


def get_tqdm_func():
    if "ipykernel" in sys.modules and "IPython" in sys.modules:
        from tqdm.notebook import tqdm as tqdm_notebook

        return tqdm_notebook
    else:
        from tqdm import tqdm as tqdm_cli

        return tqdm_cli
