from john_toolbox.preprocessing.pandas_transformers import (
    DebugTransformer,
    DropColumnsTransformer,
    EncoderTransformer,
    FunctionTransformer,
    SelectColumnsTransformer,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import numpy as np


def astype_to_string(X, column):
    return X[column].astype(str)


def fillna_value(X, column, value):
    return X[column].fillna(value)


def extract_X_y(df, target_name):
    X = df[[col for col in df.columns if col != target_name]]
    y = df[[target_name]]
    return X, y


COL_TO_KEEP = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Cabin",
    "Embarked",
]
# ----------- PIPELINE PREPROCESSING --------------
# 1. Select the appropriate columns and cast to good format.
conformity_column_list = [
    {
        "name": "DROP columns",
        "transformer": SelectColumnsTransformer(columns=COL_TO_KEEP),
    },
]

for col_to_cast in ["Embarked", "Sex", "Cabin"]:
    conformity_column_list.append(
        {
            "name": f"CAST {col_to_cast} to str",
            "transformer": FunctionTransformer(
                column=None,
                return_col=col_to_cast,
                mode="vectorized",
                func=astype_to_string,
                dict_args={"column": col_to_cast},
            ),
        }
    )

# 2. Clean Data Pipeline.
data_cleaning_list = [
    {
        "name": "Imputer_mean_Age",
        "transformer": EncoderTransformer(
            encoder=SimpleImputer,
            column="Age",
            encoder_args={"missing_values": np.nan, "strategy": "mean"},
            new_cols_prefix="Age",
            is_drop_input_col=True,
        ),
    },
    {
        "name": "Fillna by unknown",
        "transformer": FunctionTransformer(
            column=None,
            return_col="Cabin",
            mode="vectorized",
            func=fillna_value,
            dict_args={"column": "Cabin", "value": "unknown"},
        ),
    },
]

# 3. Encode categorical columns.
encoder_list = [
    {
        "name": "OHE",
        "transformer": EncoderTransformer(
            encoder=OneHotEncoder,
            column="Sex",
            encoder_args={"handle_unknown": "ignore"},
            new_cols_prefix="OHE",
            is_drop_input_col=True,
        ),
    },
    {
        "name": "OHE Embarked",
        "transformer": EncoderTransformer(
            encoder=OneHotEncoder,
            column="Embarked",
            encoder_args={"handle_unknown": "ignore", "sparse": False},
            new_cols_prefix="OHE",
            is_drop_input_col=True,
        ),
    },
    {
        "name": "OrdinalEncoder Cabin",
        "transformer": EncoderTransformer(
            encoder=OrdinalEncoder,
            column="Cabin",
            encoder_args={
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,
            },
            new_cols_prefix="Cabin",
            is_drop_input_col=True,
        ),
    },
]
