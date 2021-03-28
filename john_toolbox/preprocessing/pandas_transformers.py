import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, **transform_params):
        copy_df = X[self.columns].copy()
        return copy_df

    def fit(self, X, y=None, **fit_params):
        return self


class DebugTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.shape = None
        self.columns = None
        self.type = None

    def transform(self, X):
        print("SHAPE : ", X.shape)
        print("COLUMNS : ", X.columns)
        self.columns = X.columns
        self.type = X.dtypes

        return X

    def fit(self, X, y=None, **fit_params):
        return self


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def transform(self, X, **transform_params):
        copy_df = X.copy()
        copy_df = copy_df.drop(self.columns_to_drop, axis=1)
        return copy_df

    def fit(self, X, y=None, **fit_params):
        return self


class EncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        encoder,
        column=None,
        encoder_args=None,
        new_cols_prefix=None,
        is_drop_input_col=True,
    ):
        if encoder_args is None:
            encoder_args = {}

        self.column = column
        self.encoder = encoder(**encoder_args)
        self.new_cols_prefix = new_cols_prefix
        self.is_drop_input_col = is_drop_input_col
        if self.new_cols_prefix is None:
            self.new_cols_prefix = f"{self.column}_{self.encoder.__class__.__name__}_"

    def fit(self, X, y=None, **fit_params):
        self.encoder.fit(X[self.column])

        return self

    def transform(self, X, **transform_params):
        copy_df = X.copy()
        encoder_result_array = self.encoder.transform(copy_df[self.column])

        new_cols_size = encoder_result_array.shape[1]
        new_cols = (
            [f"{self.new_cols_prefix}_{idx}" for idx in range(new_cols_size)]
            if new_cols_size > 1
            else [self.new_cols_prefix]
        )

        encoder_result_df = pd.DataFrame(data=encoder_result_array, columns=new_cols)
        encoder_result_df.index = copy_df.index

        if (self.column != self.new_cols_prefix and self.is_drop_input_col) or (
            self.column == self.new_cols_prefix
        ):
            copy_df = copy_df.drop(self.column, axis=1)

        output_df = pd.concat([copy_df, encoder_result_df], axis=1)

        return output_df


class FunctionTransformer(BaseEstimator):
    """from
    https://stackoverflow.com/questions/42844457/scikit-learn-applying-an-arbitary-function-as-part-of-a-pipeline
    """

    def __init__(self, column, func, dict_args, return_col=None):
        self.column = column
        self.func = func
        self.dict_args = dict_args
        self.return_col = return_col

        # if None, we replace the value of the column where we apply the function
        if return_col is None:
            self.return_col = self.column

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        copy_df = X.copy()
        copy_df[self.return_col] = self.func(X[self.column], **self.dict_args)
        return copy_df
