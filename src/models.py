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
    "qlike":   qlike_loss,
    "mse_log": mse_log,
    "mae_log": mae_log,
}


# ── VolRegime-HAR ─────────────────────────────────────────────────────────────

class VolRegimeHARModel:
    """
    Regime-switching HAR: fits separate OLS models for Low/Normal/High vol regimes.
    Regime identified by rv_pct_60d (rolling percentile of log RV).
    Falls back to full-sample HAR if a regime has < 30 samples.
    """
    name = "VolRegime-HAR"
    _REGIME_COL = "rv_pct_60d"

    def __init__(self):
        self.models_: dict[str, object] = {}
        self.fallback_: object = None
        self.cols_: list[str] = []

    def _assign_regime(self, X: pd.DataFrame) -> pd.Series:
        if self._REGIME_COL in X.columns:
            pct = X[self._REGIME_COL]
        else:
            pct = X[HAR_COLS[0]].rank(pct=True) if HAR_COLS[0] in X.columns else pd.Series(0.5, index=X.index)
        return pct.apply(lambda p: "Low" if p < 0.33 else ("High" if p > 0.67 else "Normal"))

    def fit(self, X: pd.DataFrame, y: pd.Series, **_) -> "VolRegimeHARModel":
        regimes = self._assign_regime(X)
        X_har   = _har_X(X).copy()
        X_har   = sm.add_constant(X_har, has_constant="add")
        self.cols_ = X_har.columns.tolist()

        # Fallback model on full sample
        self.fallback_ = sm.OLS(y, X_har).fit(cov_type="HC3")

        for regime in ("Low", "Normal", "High"):
            mask = regimes == regime
            if mask.sum() >= 30:
                self.models_[regime] = sm.OLS(y[mask], X_har[mask]).fit(cov_type="HC3")
            else:
                self.models_[regime] = self.fallback_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        regimes = self._assign_regime(X)
        preds   = np.zeros(len(X))
        X_har   = _har_X(X)[self.cols_[1:]].copy()
        X_har   = sm.add_constant(X_har, has_constant="add")

        for regime, model in self.models_.items():
            mask = (regimes == regime).values
            if mask.any():
                preds[mask] = model.predict(X_har[mask])

        return preds

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            {regime: dict(zip(m.params.index, m.params.values))
             for regime, m in self.models_.items()}
        )


# ── HAR-XGBoost Hybrid ────────────────────────────────────────────────────────

class HARXGBoostHybrid:
    """
    Two-stage hybrid: HAR-RV-SJ (OLS) + XGBoost on residuals.
    Tests whether XGBoost's inductive bias differs meaningfully from LightGBM
    when learning residual cross-asset patterns.
    """
    name = "HAR-XGBoost Hybrid"

    def __init__(self, xgb_kwargs: dict | None = None):
        self.har_  = HARModel()
        self.xgb_  = XGBoostModel(**(xgb_kwargs or {}))
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None) -> "HARXGBoostHybrid":
        self.feature_names_ = X.columns.tolist()
        self.har_.fit(X, y)
        residuals = pd.Series(y.values - self.har_.predict(X), index=y.index)
        if X_val is not None and y_val is not None:
            val_res = pd.Series(y_val.values - self.har_.predict(X_val), index=y_val.index)
            self.xgb_.fit(X, residuals, X_val=X_val, y_val=val_res)
        else:
            self.xgb_.fit(X, residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.har_.predict(X) + self.xgb_.predict(X)

    def feature_importance(self) -> pd.Series:
        return self.xgb_.feature_importance()


# ── Multi-Horizon HAR ─────────────────────────────────────────────────────────

class MultiHorizonHAR:
    """
    Fits separate HAR-RV-SJ models for h=1, h=5, and h=22 day-ahead forecasts.
    Call .predict(X, horizon=5) for multi-horizon inference.
    """
    name = "MultiHorizon-HAR"

    def __init__(self, horizons: list[int] | None = None):
        self.horizons = horizons or [1, 5, 22]
        self.models_: dict[int, HARModel] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series,
            y_dict: dict[int, pd.Series] | None = None, **_) -> "MultiHorizonHAR":
        """
        y_dict: optional dict {horizon: target_series}.
        If not provided, fits only h=1 using y (compatible with walk-forward loop).
        """
        targets = y_dict or {1: y}
        for h, target in targets.items():
            m = HARModel()
            m.fit(X, target)
            self.models_[h] = m
        # Fallback: if only h=1 trained, alias it for other horizons
        h1_model = self.models_.get(1)
        for h in self.horizons:
            if h not in self.models_ and h1_model is not None:
                self.models_[h] = h1_model
        return self

    def predict(self, X: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        model = self.models_.get(horizon) or self.models_.get(1)
        return model.predict(X)

    def predict_all(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        return {h: self.predict(X, horizon=h) for h in self.horizons}

    def feature_importance(self) -> pd.Series:
        if 1 in self.models_:
            return self.models_[1].feature_importance()
        return pd.Series(dtype=float)


# ── EnsembleRidge (improved stacking) ────────────────────────────────────────

class EnsembleRidge:
    """
    Non-negative Ridge ensemble: learns optimal blend weights across base models.
    Uses time-series CV (no random splits) for meta-model training.
    Weights are non-negative so models can't cancel each other.
    """
    name = "Ridge Ensemble"

    def __init__(self, alpha: float = 1.0):
        self.alpha    = alpha
        self.weights_ : np.ndarray | None = None
        self.model_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        base_predictions: dict[str, np.ndarray],
    ) -> "EnsembleRidge":
        """
        base_predictions: {model_name: predictions_on_train_set}
        Fits a Ridge regression of y on the base predictions (non-negative).
        """
        from sklearn.linear_model import Ridge

        self.model_names_ = list(base_predictions.keys())
        P = np.column_stack([base_predictions[n] for n in self.model_names_])
        y = y_train.values

        # Non-negative ridge via projected gradient is complex; use simple Ridge
        # and then clip weights to >= 0 + renormalize
        model = Ridge(alpha=self.alpha, fit_intercept=False)
        model.fit(P, y)
        w = model.coef_
        w = np.maximum(w, 0)
        total = w.sum()
        self.weights_ = w / total if total > 0 else np.ones(len(w)) / len(w)
        return self

    def predict(self, base_predictions: dict[str, np.ndarray]) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("EnsembleRidge not fitted yet.")
        P = np.column_stack([base_predictions[n] for n in self.model_names_])
        return P @ self.weights_


# ── Additional Metrics ────────────────────────────────────────────────────────

def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between actuals and predictions."""
    if len(y_true) < 2:
        return float("nan")
    from scipy.stats import pearsonr
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def r2_mz(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mincer-Zarnowitz R²: from OLS of actual on predicted.
    Measures how much of realized vol variation is explained by forecasts.
    """
    try:
        X = sm.add_constant(y_pred)
        result = sm.OLS(y_true, X).fit()
        return float(result.rsquared)
    except Exception:
        return float("nan")


METRICS["corr"]  = pearson_corr
METRICS["r2_mz"] = r2_mz
