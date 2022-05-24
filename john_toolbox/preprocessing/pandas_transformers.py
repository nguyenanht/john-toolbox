from typing import Callable, Dict, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse
from john_toolbox.preprocessing.utils import compute_in_parallel

import logging

logger = logging.getLogger(__name__)


class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    This class aims to keep desired columns in Sklearn pipeline.
    See Also
    --------
    DropColumnsTransformer :  Drop columns from DataFrame.
    EncoderTransformer :  Drop columns from DataFrame.
    FunctionTransformer : Use of standard Encoder from sklearn.
    DebugTransformer : Keep track of information about DataFrame between steps.
    """

    def __init__(self, columns: List[str] = None):
        """

        Parameters
        ----------
        columns : List[str]
            List of column name to keep.
        """
        self.columns = columns

    def transform(self, X, **transform_params):
        copy_df = X[self.columns].copy()
        return copy_df

    def fit(self, X, y=None, **fit_params):
        return self


class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    This class save information between steps in sklearn pipeline and is used for debug purposes.

    See Also
    --------
    SelectColumnsTransformer : Keep columns from DataFrame.
    DropColumnsTransformer :  Drop columns from DataFrame.
    EncoderTransformer :  Drop columns from DataFrame.
    FunctionTransformer : Use of standard Encoder from sklearn.
    """

    def __init__(self):
        self.shape = None
        self.columns = None
        self.dtypes = None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"SHAPE : {X.shape}")
        logger.info(f"COLUMNS : {X.columns}")

        self.columns = X.columns
        self.dtypes = X.dtypes

        return X

    def fit(self, X, y=None, **fit_params):
        return self


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    This class let you remove a column in Sklearn pipeline.

    See Also
    --------
    SelectColumnsTransformer : Keep columns from DataFrame.
    EncoderTransformer :  Drop columns from DataFrame.
    FunctionTransformer : Use of standard Encoder from sklearn.
    DebugTransformer : Keep track of information about DataFrame between steps.
    """

    def __init__(self, columns_to_drop: List[str] = None):
        """

        Parameters
        ----------
        columns_to_drop : List[str]
            List of column to drop.
        """
        self.columns_to_drop = columns_to_drop

    def transform(self, X: pd.DataFrame, **transform_params) -> pd.DataFrame:
        copy_df = X.copy()
        copy_df = copy_df.drop(self.columns_to_drop, axis=1)
        return copy_df

    def fit(self, X, y=None, **fit_params):
        return self


class EncoderTransformer(BaseEstimator, TransformerMixin):
    """
    This class let you use standard Encoder from sklearn.

    Attributes
    ----------
    encoder :
        Standard sklearn Encoder. For example, you can provide OneHotEncoder.
    column : str, Optional
        Column to transform with the encoder.
    encoder_args : Dict, Optional
        Arguments to pass to the sklearn encoder.
    new_cols_prefix : str, Optional
        If you provide value, all generated column will have a this value as prefix.
    is_drop_input_col : bool, Optional, default True
        the old column will be removed if self.column != new_cols_prefix and is_drop_input_col == True
        or if self.column == new_cols_prefix

    See Also
    --------
    SelectColumnsTransformer : Keep columns from DataFrame.
    DropColumnsTransformer :  Drop columns from DataFrame.
    FunctionTransformer : Apply function to a column.
    DebugTransformer : Keep track of information about DataFrame between steps.

    """

    def __init__(
        self,
        encoder,
        column: str = None,
        encoder_args: Dict = None,
        new_cols_prefix: str = None,
        is_drop_input_col: bool = True,
    ):
        """

        Parameters
        ----------
        encoder :
            Standard sklearn Encoder. For example, you can provide OneHotEncoder.
        column : str, Optional
            Column to transform with the encoder.
        encoder_args : Dict, Optional
            Arguments to pass to the sklearn encoder.
        new_cols_prefix : str, Optional
            If you provide value, all generated column will have a this value as prefix.
        is_drop_input_col : bool, Optional, default True
            the old column will be removed if self.column != new_cols_prefix and is_drop_input_col == True
            or if self.column == new_cols_prefix
        """
        self.encoder_args = encoder_args
        if self.encoder_args is None:
            self.encoder_args = {}

        self.column = column
        self.encoder = encoder(**self.encoder_args)
        self.new_cols_prefix = new_cols_prefix
        self.is_drop_input_col = is_drop_input_col

        if self.new_cols_prefix is None:
            self.new_cols_prefix = (
                f"{self.column}_{self.encoder.__class__.__name__}_"
            )

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        self.encoder.fit(X[self.column].to_numpy().reshape(-1, 1))
        return self

    def transform(self, X: pd.DataFrame, **transform_params) -> pd.DataFrame:
        logger.debug(f"name encoder : {self.encoder.__class__.__name__}")
        copy_df = X.copy()

        encoder_result = self.encoder.transform(
            copy_df[self.column].to_numpy().reshape(-1, 1)
        )
        if issparse(encoder_result):
            encoder_result = encoder_result.toarray()

        logger.debug(f"SHAPE encoder_result_array : {encoder_result.shape}")

        if len(encoder_result.shape) >= 2:
            new_cols_size = encoder_result.shape[1]
        else:
            new_cols_size = 1

        try:
            new_cols = self.encoder.get_feature_names_out([self.column])

            if len(new_cols) > 1:
                # usefull only in case encoder create multiple columns like one hot encoding
                new_cols = [f"{self.new_cols_prefix}_{col}" for col in new_cols]
            logger.debug(f"new_cols = {new_cols}")

        except AttributeError:
            new_cols = (
                [
                    f"{self.new_cols_prefix}_{idx}"
                    for idx in range(new_cols_size)
                ]
                if new_cols_size > 1
                else [self.new_cols_prefix]
            )
            logger.debug(f"new_cols = {new_cols}")

        encoder_result_df = pd.DataFrame(data=encoder_result, columns=new_cols)
        encoder_result_df.index = copy_df.index

        if (self.column != self.new_cols_prefix and self.is_drop_input_col) or (
            self.column == self.new_cols_prefix
        ):
            copy_df = copy_df.drop(self.column, axis=1)

        output_df = pd.concat([copy_df, encoder_result_df], axis=1)

        return output_df


class FunctionTransformer(BaseEstimator):
    """Apply function Transformer.

    For example, please refer to : https://github.com/nguyenanht/john-toolbox/blob/develop/notebooks/tutorial1%20-%20PandasPipeline%20%26%20PandasTransformer.ipynb

    from https://stackoverflow.com/questions/42844457/scikit-learn-applying-an-arbitary-function-as-part-of-a-pipeline

    Attributes
    ----------
    column : str, Optional
        Column to transform with the encoder.
    func : Callable
        Function to apply.
    dict_args: Dict
        Arguments to pass to the function.
    mode : str, Optional, default apply_by_multiprocessing
        Mode accepted : `apply_by_multiprocessing`, `apply` or `vectorized`
        `apply_by_multiprocessing`: apply the function by using total_number of cpu minus one
        `apply`: apply in standard way the function.
        `vectorized`: vectorise an operation. For example add 2 columns.
    return_col: str, Optional, default=column
        Name of the output.
    drop_input_col: str, default=False
        Drop the input column.

    See Also
    --------
    SelectColumnsTransformer : Keep columns from DataFrame.
    DropColumnsTransformer :  Drop columns from DataFrame.
    EncoderTransformer :  Drop columns from DataFrame.
    DebugTransformer : Keep track of information about DataFrame between steps.
    """

    def __init__(
        self,
        column: str,
        func: Callable,
        dict_args: Dict,
        mode: str = "apply_by_multiprocessing",
        return_col: str = None,
        drop_input_col: bool = False,
    ):
        self.column = column
        self.func = func
        self.dict_args = dict_args
        self.return_col = return_col
        self.mode = mode
        self.drop_input_col = drop_input_col
        # if None, we replace the value of the column where we apply the function
        if return_col is None:
            self.return_col = self.column

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        copy_df = X.copy()

        if self.mode == "apply_by_multiprocessing":
            copy_df[self.return_col] = compute_in_parallel(
                series=X[self.column], func=self.func, **self.dict_args
            )
        elif self.mode == "apply":
            copy_df[self.return_col] = X[self.column].apply(
                lambda x: self.func(x, **self.dict_args)
            )
        elif self.mode == "vectorized":
            copy_df[self.return_col] = self.func(X, **self.dict_args)
        else:
            raise ValueError(
                f"{self.mode} mode not implemented. It must be in `apply_by_multiprocessing`, `apply` or `vectorized`"
            )

        if self.drop_input_col:
            copy_df.drop(self.column, axis=1)

        return copy_df
