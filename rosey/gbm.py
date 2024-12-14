# mypy: disable-error-code="import-untyped"

from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

FILEPATH_MODEL = "catboost_model_storage/catboost.cbm"


def evaluate_model(
    model: CatBoost,
    x: pd.DataFrame,
    y: pd.Series,
    metrics: List[str],
    verbose=False,
    **kwargs,
) -> Dict[str, float]:
    """
    Ingests model and processed test data then evaluates the model using the specified metrics.
    """
    test_pool = Pool(
        x,
        y,
        **kwargs,
    )
    eval_result = model.eval_metrics(  # type: ignore[attr-defined]
        test_pool,
        metrics,
    )

    # Log evaluation metrics and print if verbose
    logs = {}
    for metric in metrics:
        logs[metric] = eval_result[metric][-1]
        if verbose:
            print(f"{metric}: {eval_result[metric][-1]}")

    # Log feature importances and print if verbose
    importances = model.get_feature_importance(  # type: ignore[attr-defined]
        prettified=True
    )
    if verbose:
        print(importances)
    for name, value in zip(importances["Feature Id"], importances["Importances"]):
        logs[f"Importance of {name}"] = value

    logs["Testing Sample Count"] = len(x)
    print(logs)
    return logs


# TODO: Implement a function to write a model to a local path
def write_model(model: CatBoost, path: Path) -> None:
    model.save_model(path.cwd() / FILEPATH_MODEL)


# TODO: Implement a function to load a model from a local path
def load_model(
    path: Path, model_type: Literal["classifier", "regressor"]
) -> CatBoost:
    """
    model_type: either "classifier" or "regressor"
    """
    if model_type == "classifier":
        model = CatBoostClassifier()
    elif model_type == "regressor":
        model = CatBoostRegressor()
    else:
        raise ValueError("model_type must be either 'classifier' or 'regressor'")

    # join the path to the filename
    model_path = path.cwd() / FILEPATH_MODEL

    return model.load_model(model_path)


class _BoostingModelTrainer(BaseEstimator):
    """Base class for CatBoost Models Trainers"""

    def __init__(
        self,
        iterations: int,
        cat_features: Optional[List[str]],
        text_features: Optional[List[str]],
        embedding_features: Optional[List[str]],
        early_stopping_rounds: int,
        use_best_model: bool,
        verbose: bool,
        random_state: int,
        loss_function: str,
        eval_metric: str,
    ):
        self.iterations = iterations
        self.cat_features = cat_features
        self.text_features = text_features
        self.embedding_features = embedding_features
        self.early_stopping_rounds = early_stopping_rounds
        self.use_best_model = use_best_model
        self.verbose = verbose
        self.random_state = random_state
        self.loss_function = loss_function
        self.eval_metric = eval_metric

        self.model_ = None

    @property
    def get_model(self):
        if self.model_:
            return self.model_
        raise NotFittedError("You must call `.fit()` first")

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        stratify_by: Optional[pd.DataFrame] = None,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> CatBoost:
        """
        Ingest processed train data, perform train test split, and fit
        the model .
        """
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        if x_val is None or y_val is None:
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, stratify=stratify_by
            )
        else:
            x_train, y_train = x, y

        self.model_ = self.model_.fit(  # type: ignore[attr-defined]
            x_train,
            y_train,
            eval_set=(x_val, y_val),
            **kwargs,
        )

        # TODO (2024/07/03) @srose: Implement batched fitting if required

        print(
            {
                "Iterations Completed": min(
                    (
                        self.model_.best_iteration_  # type: ignore[attr-defined]
                        + self.early_stopping_rounds
                        + 1
                    ),
                    self.iterations,
                ),
                "Training Sample Count": len(x_train),
            }
        )
        return self.model_

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(x)  # type: ignore[attr-defined]


class CBClassifierTrainer(_BoostingModelTrainer):
    """
    CatBoost Classifier Trainer.

    NOTE: this class is set up for binary classification
    """

    def __init__(
        self,
        iterations=10_000,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        early_stopping_rounds=25,
        use_best_model=True,
        verbose=True,
        random_state=42,
        loss_function="Logloss",
        eval_metric="AUC",  # TODO: Does this trigger early stopping or Loss Function?
        **kwargs,
    ):
        super().__init__(
            iterations,
            cat_features,
            text_features,
            embedding_features,
            early_stopping_rounds,
            use_best_model,
            verbose,
            random_state,
            loss_function,
            eval_metric,
        )

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            cat_features=self.cat_features,
            text_features=self.text_features,
            embedding_features=self.embedding_features,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=self.use_best_model,
            verbose=self.verbose,
            random_state=self.random_state,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            **kwargs,
        )


class CBRegressorTrainer(_BoostingModelTrainer):
    """CatBoost Regressor Trainer."""

    def __init__(
        self,
        iterations=10_000,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        early_stopping_rounds=25,
        use_best_model=True,
        verbose=True,
        random_state=42,
        loss_function="RMSE",
        eval_metric="RMSE",
        **kwargs,
    ):
        super().__init__(
            iterations,
            cat_features,
            text_features,
            embedding_features,
            early_stopping_rounds,
            use_best_model,
            verbose,
            random_state,
            loss_function,
            eval_metric,
        )

        self.eval_metric = eval_metric

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            cat_features=self.cat_features,
            text_features=self.text_features,
            embedding_features=self.embedding_features,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=self.use_best_model,
            verbose=self.verbose,
            random_state=self.random_state,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            **kwargs,
        )
        