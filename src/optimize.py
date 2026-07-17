"""
Hyperparameter optimization for the HAR-LGBM Hybrid using Optuna.

Strategy:
  - Tune once on an expanding window from the first 60% of data
  - Apply the best parameters to all subsequent walk-forward folds
  - This avoids the combinatorial cost of re-tuning every fold while
    still finding meaningful improvements over defaults

Also implements:
  - StackingEnsemble: Ridge meta-learner on top of all base model OOF predictions
  - Realized semi-variance features (upside/downside RV)
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── Realized Semi-Variances ───────────────────────────────────────────────────

def realized_semi_variances(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose realized variance into upside (RS+) and downside (RS-) components.
    Patton & Sheppard (2015): signed semi-variances predict future vol asymmetrically.
    Negative semi-variance is more predictive of future volatility than positive.
    """
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    pos = df.groupby("date")["log_ret"].apply(
        lambda r: np.sqrt((r[r > 0] ** 2).sum()) * np.sqrt(365)
    ).rename("rv_pos")
    neg = df.groupby("date")["log_ret"].apply(
        lambda r: np.sqrt((r[r < 0] ** 2).sum()) * np.sqrt(365)
    ).rename("rv_neg")

    pos.index = pd.to_datetime(pos.index, utc=True)
    neg.index = pd.to_datetime(neg.index, utc=True)

    log_pos = np.log(pos.clip(lower=1e-8)).shift(1).rename("log_rv_pos_lag1")
    log_neg = np.log(neg.clip(lower=1e-8)).shift(1).rename("log_rv_neg_lag1")

    # 5-day rolling averages
    log_pos_5d = np.log(pos.clip(lower=1e-8)).shift(1).rolling(5).mean().rename("log_rv_pos_5d")
    log_neg_5d = np.log(neg.clip(lower=1e-8)).shift(1).rolling(5).mean().rename("log_rv_neg_5d")

    # Asymmetry ratio: how much of recent vol was downside-driven
    total = pos + neg
    neg_share = (neg / total.clip(lower=1e-8)).shift(1).rename("neg_rv_share")

    return pd.DataFrame([log_pos, log_neg, log_pos_5d, log_neg_5d, neg_share]).T


# ── Optuna Tuning ─────────────────────────────────────────────────────────────

def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 60,
    cv_folds: int = 5,
) -> dict:
    """
    Tune LightGBM hyperparameters using Optuna with time-series cross-validation.
    Returns the best parameter dict found.

    Uses walk-forward CV internally (not k-fold) to avoid lookahead.
    """
    n = len(X_train)
    fold_size = n // (cv_folds + 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        mse_scores = []
        for fold in range(1, cv_folds + 1):
            train_end = fold * fold_size
            val_start = train_end + 5   # 5-day embargo
            val_end   = val_start + fold_size
            if val_end > n:
                continue

            X_tr = X_train.iloc[:train_end].fillna(0)
            y_tr = y_train.iloc[:train_end]
            X_val = X_train.iloc[val_start:val_end].fillna(0)
            y_val = y_train.iloc[val_start:val_end]

            import lightgbm as lgb_inner
            model = lgb_inner.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb_inner.early_stopping(30, verbose=False),
                           lgb_inner.log_evaluation(-1)],
            )
            pred = model.predict(X_val)
            mse_scores.append(np.mean((y_val.values - pred) ** 2))

        return np.mean(mse_scores) if mse_scores else 1e6

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  Optuna best MSE(log): {study.best_value:.5f}  (trials: {n_trials})")
    print(f"  Best params: {best}")
    return best


# ── Tuned HAR-LGBM Hybrid ─────────────────────────────────────────────────────

class HARLGBMTuned:
    """
    HAR-LGBM Hybrid with Optuna-tuned hyperparameters.
    Tunes once on the first `tune_on_pct` fraction of training data,
    then applies fixed params to all subsequent walk-forward refits.
    """
    name = "HAR-LGBM Tuned"

    def __init__(self, n_trials: int = 60, tune_on_pct: float = 0.6):
        self.n_trials = n_trials
        self.tune_on_pct = tune_on_pct
        self.best_params_ = None
        self._tuned = False
        self.har_ = None
        self.lgbm_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val=None, y_val=None) -> "HARLGBMTuned":
        from models import HARModel
        import lightgbm as lgb_m

        har_cols = [c for c in ["rv_1d","rv_5d","rv_22d","jump_pos","jump_neg"] if c in X.columns]

        # Stage 1: HAR
        self.har_ = HARModel()
        self.har_.fit(X, y)
        har_pred = self.har_.predict(X)
        residuals = pd.Series(y.values - har_pred, index=y.index)

        # Tune on first tune_on_pct of data (once only)
        if not self._tuned:
            cut = int(len(X) * self.tune_on_pct)
            print(f"  Tuning LGBM on first {cut} rows ({self.n_trials} trials)...")
            self.best_params_ = tune_lgbm(
                X.iloc[:cut], residuals.iloc[:cut],
                n_trials=self.n_trials,
            )
            self._tuned = True

        # Stage 2: LGBM on residuals with tuned params
        params = {**self.best_params_, "verbose": -1, "n_jobs": -1, "random_state": 42}
        train_data = lgb_m.Dataset(X.fillna(0), label=residuals)
        callbacks = [lgb_m.log_evaluation(-1)]

        if X_val is not None and y_val is not None:
            har_val_pred = self.har_.predict(X_val)
            val_res = pd.Series(y_val.values - har_val_pred, index=y_val.index)
            val_data = lgb_m.Dataset(X_val.fillna(0), label=val_res, reference=train_data)
            callbacks.append(lgb_m.early_stopping(50, verbose=False))
            self.lgbm_ = lgb_m.train(params, train_data, valid_sets=[val_data], callbacks=callbacks)
        else:
            self.lgbm_ = lgb_m.train(params, train_data, callbacks=callbacks)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.har_.predict(X) + self.lgbm_.predict(X.fillna(0))

    def feature_importance(self) -> pd.Series:
        imp = self.lgbm_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=[f"f{i}" for i in range(len(imp))]).sort_values(ascending=False)


# ── Stacking Ensemble ─────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Meta-learner that combines out-of-fold predictions from all base models.

    Stage 1: Each base model generates OOF predictions on the training set.
    Stage 2: Ridge regression is trained on the OOF predictions → learns
             optimal blending weights per model per regime.
    Inference: base model predictions → Ridge → final forecast.

    This exploits the fact that different models excel in different regimes.
    """
    name = "Stacking Ensemble"

    def __init__(self, base_model_factories: dict, n_folds: int = 5):
        self.factories = base_model_factories
        self.n_folds = n_folds
        self.meta_ = None
        self.scaler_ = StandardScaler()
        self.base_models_ = {}   # fitted on full training set

    def _oof_predictions(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Generate out-of-fold predictions for each base model."""
        n = len(X)
        fold_size = n // (self.n_folds + 1)
        oof = pd.DataFrame(index=X.index, columns=list(self.factories.keys()), dtype=float)

        for fold in range(1, self.n_folds + 1):
            train_end = fold * fold_size
            val_start = train_end + 5
            val_end   = min(val_start + fold_size, n)
            if val_end > n: continue

            X_tr = X.iloc[:train_end]
            y_tr = y.iloc[:train_end]
            X_val = X.iloc[val_start:val_end]

            for name, factory in self.factories.items():
                m = factory()
                try:
                    m.fit(X_tr, y_tr)
                except Exception:
                    m.fit(X_tr, y_tr)
                oof.loc[X_val.index, name] = m.predict(X_val)

        return oof.dropna()

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val=None, y_val=None) -> "StackingEnsemble":
        print("  Generating OOF predictions for stacking...")
        oof = self._oof_predictions(X, y)
        y_oof = y.loc[oof.index]

        # Meta-learner: Ridge on OOF predictions
        X_meta = self.scaler_.fit_transform(oof.values)
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        self.meta_ = RidgeCV(alphas=alphas, cv=5).fit(X_meta, y_oof)
        print(f"  Meta-learner weights: { {n: f'{w:.3f}' for n,w in zip(self.factories, self.meta_.coef_)} }")

        # Fit base models on full training data
        for name, factory in self.factories.items():
            m = factory()
            try:
                m.fit(X, y)
            except Exception:
                m.fit(X, y)
            self.base_models_[name] = m

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([
            self.base_models_[name].predict(X)
            for name in self.factories
        ])
        X_meta = self.scaler_.transform(base_preds)
        return self.meta_.predict(X_meta)
