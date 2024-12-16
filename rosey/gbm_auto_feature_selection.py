"""
Implemenent an automated feature seleection pipeline for CatBoost models

INPUTS:
    - pd.DataFrame of raw data
    - target variable name
    - model type (classifier or regressor)

OUTPUT:
    - dictionary of feature name, feature importance, and feature type

This process will include the following:
    . Set assumptions around:
        - What data types are feasible
        - how much of a feature can be null
        - Correlation threshold
    . Ingest a pandas DataFrame of raw data
    . Peform data cleaning/prep using generalized pipeline based on feature Dtypes
        - Define strategy for processing text and categorical variables
    . Drop features based on null data fraction
    . Drop features based on correlation threshold
    . Train a CatBoost model
    . Use catboost.select_features() method
"""

import pandas as pd # type: ignore
from gbm import CBClassifierTrainer, CBRegressorTrainer # type: ignore

# TODO: Build out complete list of Pandas Dtypes
FEASIBLE_DTYPES = [
    "int64",
    "float64",
    "bool",
    "datetime64",
    "object"
    ]

# TODO: Is there a general rule of thumb for this threshold?
NULL_DATA_FRAC_THRESHOLD = 0.5

# TODO: Find basis for assumption
UNIQUE_VALUE_THRESHOLD = 0.9

# TODO: Find basis for assumption
HIGH_CORR_THRESHOLD = 0.95

FILL_NULL_DICT = {
    "int64": pd.NA, # TODO: Will pd.NA work here?
    "float64": pd.NA, # TODO: Will pd.NA work here?
    "object": "<unknown>",
    "string": "<unknown>",
    "date": "1900/1/1", # TODO: Should this be a date in the far past?
    "bool": None,
}

# NOTE: MAX Number of features to keep by end of feature selection process
TOP_K_FEATURES = 10

def _null_data_fraction_filter(df: pd.DataFrame, column_name: str) -> tuple[pd.DataFrame, str]:
    """
    Remove features from a DataFrame that have a null data fraction above the threshold
    """

    column_state = "kept"
    null_data_frac = df[column_name].isnull().sum() / len(df)

    if null_data_frac > NULL_DATA_FRAC_THRESHOLD:
        df.drop(columns=[column_name], inplace=True)
        column_state = "dropped"

    return df, column_state

def _process_string(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Process string data types in a DataFrame
    Cast to python str type
    Strip whitespace
    Convert to lowercase
    Fill null values
    """

    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    return df[column_name].fillna(FILL_NULL_DICT["string"], inplace=True)

def _process_numerics(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Process numeric data types in a DataFrame
    Cast to float
    Round to 2 decimal places
    """

    df[column_name] = df[column_name].astype(float)
    return df[column_name].round(2)

def _process_date(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Process date data types in a DataFrame
    Cast to datetime
    Fill null values
    """

    df['date_column'] = pd.to_datetime(df['date_column'], infer_datetime_format=True)
    
    # NOTE: CatBoost models cannot handle datetime data types, so cast to String
    return df['date_column'].dt.strftime('%Y-%m-%d')

def _process_bool(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Process boolean data types in a DataFrame by casting to float
        TRUE -> 1
        FALSE -> 0
    """
    return df[column_name].astype(float)

def _unique_value_filter(df: pd.DataFrame, column_name: str) -> tuple[pd.DataFrame, str]:
    """
    Remove features from a DataFrame that have a high frequency of unique values
    """

    column_state = "kept"
    unique_value_frac = df[column_name].nunique() / len(df)

    if unique_value_frac > UNIQUE_VALUE_THRESHOLD:
        df.drop(columns=[column_name], inplace=True)
        column_state = "dropped"

    return df, column_state

def _remove_numeric_outliers(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on IQR method

    PARAMS:
        - DataFrame
        - Column name to process
    RETURNS:
        - DataFrame with outliers removed
    """

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column_name] > lower_bound) & (df[column_name] < upper_bound)]

def _catboost_data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw data for CatBoost model training
    """

    for column in df.columns:

        df, column_state = _null_data_fraction_filter(df, column)
        if column_state == "dropped":
            print(f"Column {column} dropped due to high null data fraction")
            continue

        # TODO: replace with dictionary of functions?
        if df[column].dtype == "object":
            df = _process_string(df, column)

        elif df[column].dtype == "float64":
            df = _process_numerics(df, column)
            df = _remove_numeric_outliers(df, column)

        elif df[column].dtype == "int64":
            df = _process_numerics(df, column)
            df = _remove_numeric_outliers(df, column)

        elif df[column].dtype == "datetime64":
            df = _process_date(df, column)

        elif df[column].dtype == "bool":
            df = _process_bool(df, column)
        else:
            print(f"Data type {df[column].dtype} is not supported")

        df, column_state = _unique_value_filter(df, column)
        if column_state == "dropped":
            print(f"Column {column} dropped due to high unique value frequency")
            continue

    return df

# TODO: Design high correlation feature removal function

def main(df: pd.DataFrame, target_column: str,  model_type: str) -> dict:
    """
    Main function to run the automated feature selection pipeline
    """

    df = _catboost_data_preprocessing(df)

    if model_type == "classifier":
        trainer = CBClassifierTrainer()
    elif model_type == "regressor":
        trainer = CBRegressorTrainer()
    else:
        raise ValueError("Model type must be either 'classifier' or 'regressor'")

    # TODO: Implement model.select_features() method
    model = trainer.fit(df.drop(columns=[target_column]), df[target_column])

    return model.select_features() 

if __name__ == "__main__":

    # Test Input Data
    test_data = {
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [1, 2, 3, 4, 5],
        "feature_3": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(test_data)

    model_type = "classifier"
    target_variable = "target"

    print("Done")