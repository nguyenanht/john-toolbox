from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline

import logging

LOGGER = logging.getLogger(__name__)


class PandasPipeline:
    def __init__(self, steps: List, target_name: str, verbose: bool = True):
        self.target_name = target_name
        self.columns = None
        self._some_target_modality = None

        self.feature_processing = Pipeline(
            steps=steps,
            verbose=verbose,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns = df.columns
        self._some_target_modality = df[self.target_name].iat[
            0
        ]  # in case we do not have the target for the prediction
        return self.feature_processing.fit_transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.target_name not in df.columns:
            df[self.target_name] = [self._some_target_modality] * len(df)

        df = df[self.columns]  # same order as df_train

        return self.feature_processing.transform(df)
