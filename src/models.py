"""
Volatility forecasting models.

Implements:
  - HAR-RV-SJ  (econometric baseline — OLS on HAR components + signed jumps)
  - HAR-Lasso  (regularized HAR with all features)
  - XGBoost    (nonlinear, full feature set)
  - LightGBM   (fast gradient boosting, full feature set)
  - HAR-LGBM   (hybrid: HAR residual fitted by LightGBM)

All models expose a common interface:
  .fit(X_train, y_train)
  .predict(X_test) -> np.ndarray
  .feature_importance() -> pd.Series   (where applicable)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ── Shared utilities ──────────────────────────────────────────────────────────

HAR_COLS = ["rv_1d", "rv_5d", "rv_22d", "jump_pos", "jump_neg"]


def _har_X(X: pd.DataFrame) -> pd.DataFrame:
    """Extract HAR columns that are present in X."""
    cols = [c for c in HAR_COLS if c in X.columns]
    return X[cols]


# ── HAR-RV-SJ (OLS) ──────────────────────────────────────────────────────────

class HARModel:
    """
    Standard HAR-RV-SJ model estimated by OLS.
    Predicts log(RV_t+h) from HAR components and signed jumps.
    Uses statsmodels for coefficient inference.
    """
    name = "HAR-RV-SJ"

    def __init__(self):
        self.result_ = None
        self.cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        X_har = _har_X(X).copy()
        X_har = sm.add_constant(X_har, has_constant="add")
        self.cols_ = X_har.columns.tolist()
        self.result_ = sm.OLS(y, X_har).fit(cov_type="HC3")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_har = _har_X(X)[self.cols_[1:]].copy()  # drop const
        X_har = sm.add_constant(X_har, has_constant="add")
        return self.result_.predict(X_har).values

    def summary(self) -> str:
        return str(self.result_.summary())

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            np.abs(self.result_.params),
            index=self.result_.params.index,
            name="coef_abs"
        ).drop("const", errors="ignore")


# ── HAR-Lasso ─────────────────────────────────────────────────────────────────

class HARLassoModel:
    """
    Lasso-regularized HAR extended with full feature set.
    Lambda selected by time-series cross-validation.
    """
    name = "HAR-Lasso"

    def __init__(self, cv: int = 5, max_iter: int = 10000):
        self.cv = cv
        self.max_iter = max_iter
        self.pipeline_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARLassoModel":
        self.feature_names_ = X.columns.tolist()
        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=self.cv, max_iter=self.max_iter, n_jobs=-1)),
        ])
        self.pipeline_.fit(X.fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(X.fillna(0))

    def feature_importance(self) -> pd.Series:
        coefs = self.pipeline_.named_steps["lasso"].coef_
        return pd.Series(
            np.abs(coefs), index=self.feature_names_, name="coef_abs"
        ).sort_values(ascending=False)


# ── XGBoost ───────────────────────────────────────────────────────────────────

class XGBoostModel:
    """XGBoost regressor with QLIKE-approximating objective (log-space MSE)."""
    name = "XGBoost"

    def __init__(self, **kwargs):
        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(kwargs)
        self.model_ = xgb.XGBRegressor(**defaults)
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "XGBoostModel":
        self.feature_names_ = X.columns.tolist()
        eval_set = [(X.fillna(0), y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val.fillna(0), y_val))
        self.model_.fit(
            X.fillna(0), y,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.fillna(0))

    def feature_importance(self) -> pd.Series:
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self.feature_names_, name="importance").sort_values(ascending=False)


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LGBMModel:
    """LightGBM regressor — faster than XGBoost, often competitive."""
    name = "LightGBM"

    def __init__(self, **kwargs):
        defaults = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(kwargs)
        self.params_ = defaults
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "LGBMModel":
        self.feature_names_ = X.columns.tolist()
        train_data = lgb.Dataset(X.fillna(0), label=y)
        callbacks = [lgb.log_evaluation(period=-1)]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val.fillna(0), label=y_val, reference=train_data)
            callbacks.append(lgb.early_stopping(50, verbose=False))
            self.model_ = lgb.train(
                self.params_, train_data,
                valid_sets=[val_data],
                callbacks=callbacks,
            )
        else:
            self.model_ = lgb.train(self.params_, train_data, callbacks=callbacks)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X.fillna(0))

    def feature_importance(self) -> pd.Series:
        imp = self.model_.feature_importance(importance_type="gain")
        return pd.Series(
            imp, index=self.feature_names_, name="importance"
        ).sort_values(ascending=False)


# ── HAR-LGBM Hybrid ───────────────────────────────────────────────────────────

class HARLGBMHybrid:
    """
    Two-stage hybrid model:
      Stage 1: HAR-RV-SJ fitted by OLS → captures linear structure
      Stage 2: LightGBM fitted on HAR residuals → captures nonlinear/cross-asset effects
    Final prediction = HAR prediction + LGBM(residual)
    """
    name = "HAR-LGBM Hybrid"

    def __init__(self, lgbm_kwargs: dict | None = None):
        self.har_ = HARModel()
        self.lgbm_ = LGBMModel(**(lgbm_kwargs or {}))
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "HARLGBMHybrid":
        self.feature_names_ = X.columns.tolist()

        # Stage 1: HAR
        self.har_.fit(X, y)
        har_pred = self.har_.predict(X)
        residuals = y.values - har_pred

        # Stage 2: LGBM on residuals, full feature set
        res_series = pd.Series(residuals, index=y.index)
        if X_val is not None and y_val is not None:
            har_val_pred = self.har_.predict(X_val)
            val_residuals = pd.Series(y_val.values - har_val_pred, index=y_val.index)
            self.lgbm_.fit(X, res_series, X_val=X_val, y_val=val_residuals)
        else:
            self.lgbm_.fit(X, res_series)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.har_.predict(X) + self.lgbm_.predict(X)

    def feature_importance(self) -> pd.Series:
        return self.lgbm_.feature_importance()


# ── Loss Functions ────────────────────────────────────────────────────────────

def qlike_loss(y_true: np.ndarray, y_pred_log: np.ndarray) -> float:
    """
    QLIKE loss for volatility forecasts (operates on log-RV predictions).
    QLIKE = mean( RV/sigma^2 - log(RV/sigma^2) - 1 )
    Robust to outliers; standard in MCS comparisons.
    """
    rv_true = np.exp(y_true)
    sigma2_pred = np.exp(y_pred_log)
    ratio = rv_true / sigma2_pred
    return float(np.mean(ratio - np.log(ratio) - 1))


def mse_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


METRICS = {
    "qlike": qlike_loss,
    "mse_log": mse_log,
    "mae_log": mae_log,
}
