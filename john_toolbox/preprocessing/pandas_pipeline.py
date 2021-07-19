from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline

import logging

LOGGER = logging.getLogger(__name__)


class PandasPipeline:
    """
    Wrapper for handling pandas in sklearn pipeline.
    Instead of returning a numpy array, it returns Pandas DataFrame with all columns name.

    Attributes
    ----------
    target_name : str
                target column name
    columns : List
        family name of the person

    Methods
    -------
        fit_transform(df):
            Fit all transformers and transforms DataFrame.
        transform(df):
            Transforms DataFrame with the fitted Pipeline.
    """

    def __init__(self, steps: List, target_name: str, verbose: bool = True):
        """

        Parameters
        ----------
        steps : List[(name, john_toolbox.preprocessing.pandas_transformer)
        or List[{"name": name, "transformer": john_toolbox.preprocessing.pandas_transformer}]
            List of tuple or dict with name of the step and pandas_transformer.
        target_name : str
            Column name of the target.
        verbose :bool
            Display time execution for each steps.

        Examples
        --------
        How to use PandasPipeline and pandas_transformers.
        >>> from john_toolbox.preprocessing.pandas_transformers import DropColumnsTransformer
        >>> step_list = [("drop_column", DropColumnsTransformer(columns_to_drop=["target_name"]))]
        >>> PandasPipeline(steps=step_list, target_name="target_name", verbose=True)
        """
        self.target_name = target_name
        self.columns = []
        self._some_target_modality = None

        if isinstance(steps[0], dict):
            steps = [(elem["name"], elem["transformer"]) for elem in steps]

        self.sklearn_pipeline = Pipeline(
            steps=steps,
            verbose=verbose,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns = list(df.columns)
        self._some_target_modality = df[self.target_name].iat[
            0
        ]  # in case we do not have the target for the prediction
        return self.sklearn_pipeline.fit_transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.target_name not in df.columns:
            df[self.target_name] = [self._some_target_modality] * len(df)

        df = df[self.columns]  # same order as df_train

        return self.sklearn_pipeline.transform(df)
