from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .features import build_structured_preprocessor


@dataclass
class RegressionResult:
    model_name: str
    rmse: float
    mae: float
    r2: float
    fitted_pipeline: Pipeline


def train_multiple_regression_models(
    df: pd.DataFrame,
    target_col: str = "profit",
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, RegressionResult]:
    """
    Train multiple regression models on structured features.
    Designed for quick experimentation + SHAP compatibility.
    """
    # Basic feature set (expand in notebooks)
    numeric_cols = ["price", "num_subscribers", "num_reviews", "num_lectures", "year", "month", "day", "content_duration_hours"]
    categorical_cols = ["subject", "level", "is_paid"]

    X = df[numeric_cols + categorical_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    pre = build_structured_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    candidates: Dict[str, RegressorMixin] = {
        "ridge": Ridge(alpha=5.0, random_state=seed),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gbdt": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_depth=None,
            max_iter=500,
            random_state=seed,
        ),
    }

    results: Dict[str, RegressionResult] = {}
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        results[name] = RegressionResult(
            model_name=name,
            rmse=rmse,
            mae=mae,
            r2=r2,
            fitted_pipeline=pipe,
        )

    return results
