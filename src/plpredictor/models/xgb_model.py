"""XGBoost H/D/A classifier with probability calibration and season recency weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from plpredictor import config
from plpredictor.evaluation.metrics import outcome_index

XGB_PARAMS = {
    "n_estimators": 250, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
    "colsample_bytree": 0.7, "min_child_weight": 8, "reg_lambda": 5.0,
    "objective": "multi:softprob", "eval_metric": "mlogloss", "tree_method": "hist",
    "random_state": config.SEED, "n_jobs": 4,
}  # fmt: skip


def season_weights(train: pd.DataFrame, ref_date: pd.Timestamp, decay: float) -> np.ndarray:
    """Weight = decay ** (seasons back); decay=1 means unweighted."""
    if decay == 1.0:
        return np.ones(len(train))
    ref = config.season_start_year(ref_date.date())
    back = ref - train["season"].str[:4].astype(int).to_numpy()
    return decay ** np.clip(back, 0, None)


class XGBModel:
    """Fit on past matches. With ``calibrate`` the newest 20% of rows fit a Platt-style
    (multinomial logistic on log-probabilities) calibrator, so probabilities are honest."""

    def __init__(self, features: list[str], season_decay: float = 0.8, calibrate: bool = True):
        self.features, self.season_decay, self.calibrate = features, season_decay, calibrate
        self.model: XGBClassifier | None = None
        self.cal: LogisticRegression | None = None

    def fit(self, train: pd.DataFrame, ref_date: pd.Timestamp) -> XGBModel:
        train = train.sort_values("date")
        y = outcome_index(train["result"])
        w = season_weights(train, ref_date, self.season_decay)
        X = train[self.features]
        cut = int(len(train) * 0.8) if self.calibrate else len(train)
        self.model = XGBClassifier(**XGB_PARAMS).fit(X.iloc[:cut], y[:cut], sample_weight=w[:cut])
        if self.calibrate and len(train) - cut >= 100:
            raw = self.model.predict_proba(X.iloc[cut:])
            self.cal = LogisticRegression(C=10.0, max_iter=500).fit(
                np.log(np.clip(raw, 1e-6, 1)), y[cut:]
            )
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(test[self.features])
        if self.cal is None:
            return raw
        return self.cal.predict_proba(np.log(np.clip(raw, 1e-6, 1)))

    def importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(
            ascending=False
        )
