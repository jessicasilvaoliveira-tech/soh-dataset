"""Treinamento e avaliação honesta dos regressores.

Protocolo:
1. Split holdout estratificado por faixa de SOH e agrupado por battery_id.
2. Validação cruzada agrupada repetida apenas no conjunto de treino.
3. Seleção do modelo pela média da CV (não pelo teste).
4. Avaliação de todos os modelos na validação (diagnóstico).
5. Refit do modelo selecionado em treino+validação e avaliação única no teste.
6. Métricas por linha e por bateria, intervalos bootstrap e análise de resíduos.
"""
import copy
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from config import (
    CV_FOLDS,
    CV_REPEATS,
    N_BOOTSTRAP,
    OUTPUT_DIR,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGET = "target_soh"

PULSE_FEATURES = [
    "pulse_ocv_v",
    "pulse_v_loaded_v",
    "pulse_delta_v",
    "pulse_current_a",
    "pulse_r_dynamic_mohm",
    "pulse_r_early_mohm",
    "pulse_v_recovery_v",
    "pulse_recovery_delta_v",
    "pulse_relaxation_ratio",
    "pulse_tau_recovery_s",
    "soc_pct",
    "temperature_c",
]

CRANK_FEATURES = [
    "crank_ocv_v",
    "crank_v_min_v",
    "crank_v_mean_v",
    "crank_delta_v_min_v",
    "crank_i_max_a",
    "crank_i_mean_a",
    "crank_r_eff_mohm",
    "crank_t_v_min_s",
    "crank_v_end_v",
    "crank_v_recovery_v",
    "crank_recovery_gain_v",
    "crank_tau_recovery_s",
    "soc_pct",
    "temperature_c",
]

COMBINED_FEATURES = PULSE_FEATURES + [
    c for c in CRANK_FEATURES if c not in PULSE_FEATURES
]

FORBIDDEN_FEATURES = {
    TARGET,
    "capacity_ah_simulated",
    "capacity_ah_measured",
    "capacity_ah_parameter",
    "soh_capacity_parameter",
    "resistance_health",
    "battery_id",
    "scenario_id",
}


def assert_no_leakage(feature_cols):
    bad = sorted(set(feature_cols) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(f"Features proibidas/vazadas: {bad}")


class GroupEarlyStoppingMLP(BaseEstimator, RegressorMixin):
    """MLP com early stopping usando grupos (baterias) para validação interna."""

    def __init__(
        self,
        hidden_layer_sizes=(64,),
        alpha=0.03,
        learning_rate_init=0.002,
        max_epochs=1000,
        patience=60,
        validation_fraction=0.20,
        random_state=SEED,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _split_groups(self, X, y, groups):
        rng = np.random.default_rng(self.random_state)
        if groups is None:
            idx = rng.permutation(len(X))
            n_val = max(1, int(round(len(X) * self.validation_fraction)))
            return idx[n_val:], idx[:n_val]

        groups = np.asarray(groups)
        unique = np.unique(groups)
        rng.shuffle(unique)
        n_val = max(1, int(round(len(unique) * self.validation_fraction)))
        val_groups = unique[:n_val]
        val_mask = np.isin(groups, val_groups)
        return np.where(~val_mask)[0], np.where(val_mask)[0]

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        tr_idx, val_idx = self._split_groups(X, y, groups)
        if len(tr_idx) < 10 or len(val_idx) < 3:
            tr_idx = np.arange(len(X))
            val_idx = np.arange(len(X))

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        self.y_mean_ = float(np.mean(y_tr))
        self.y_std_ = float(np.std(y_tr))
        if self.y_std_ < 1e-9:
            self.y_std_ = 1.0

        y_tr_scaled = (y_tr - self.y_mean_) / self.y_std_
        y_val_scaled = (y_val - self.y_mean_) / self.y_std_

        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=1,
            warm_start=True,
            early_stopping=False,
            random_state=self.random_state,
        )

        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0
        epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model_.fit(X_tr, y_tr_scaled)
            pred_val = self.model_.predict(X_val)
            loss = mean_squared_error(y_val_scaled, pred_val)

            if loss < best_loss - 1e-7:
                best_loss = loss
                best_state = (
                    [coef.copy() for coef in self.model_.coefs_],
                    [intercept.copy() for intercept in self.model_.intercepts_],
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model_.coefs_ = best_state[0]
            self.model_.intercepts_ = best_state[1]

        self.n_iter_ = epoch
        self.best_validation_loss_ = float(best_loss)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.model_.predict(X) * self.y_std_ + self.y_mean_


class PhysicalResistanceTransformer(BaseEstimator, TransformerMixin):
    """Baseline físico: resistência corrigida por temperatura e SOC.

    Usa os coeficientes exploratórios do simulador como um baseline oracle.
    Serve para medir quanto do desempenho é explicado pela relação R-SOH.
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(np.asarray(X), columns=self.feature_names)
        cold = np.exp(0.018 * (25.0 - df["temperature_c"].to_numpy()))
        low_soc = 1.0 + 0.45 * (100.0 - df["soc_pct"].to_numpy()) / 100.0

        cols = []
        if "pulse_r_dynamic_mohm" in df.columns:
            cols.append(
                df["pulse_r_dynamic_mohm"].to_numpy() / (cold * low_soc)
            )
        if "crank_r_eff_mohm" in df.columns:
            cols.append(
                df["crank_r_eff_mohm"].to_numpy() / (cold * low_soc)
            )
        if not cols:
            raise ValueError("Nenhuma feature de resistência encontrada.")
        return np.column_stack(cols)


def make_models(feature_cols, seed=SEED):
    feature_cols = tuple(feature_cols)
    models = {
        "DummyMean": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "PhysicalRidge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("physical", PhysicalResistanceTransformer(feature_names=feature_cols)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=5,
                max_features=0.7,
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
        "SVR_RBF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(C=30.0, epsilon=0.5, gamma="scale")),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            )),
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GroupEarlyStoppingMLP(
                hidden_layer_sizes=(64,),
                alpha=0.03,
                learning_rate_init=0.002,
                max_epochs=1000,
                patience=60,
                validation_fraction=0.20,
                random_state=seed,
            )),
        ]),
    }
    for k in (5, 15, 30):
        models[f"KNN_k{k}"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=k,
                weights="distance",
                n_jobs=-1,
            )),
        ])

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=seed,
            )),
        ])
    except ImportError:
        print("XGBoost não instalado; ignorando.")

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=7,
                max_depth=3,
                reg_lambda=5.0,
                verbosity=-1,
                random_state=seed,
            )),
        ])
    except ImportError:
        print("LightGBM não instalado; ignorando.")

    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(
                iterations=500,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=5.0,
                loss_function="RMSE",
                random_seed=seed,
                thread_count=-1,
                verbose=0,
                allow_writing_files=False,
            )),
        ])
    except ImportError:
        print("CatBoost não instalado; ignorando.")
    return models


def fit_model(model, model_name, X, y, groups):
    if model_name == "MLP":
        model.fit(X, y, model__groups=groups)
    else:
        model.fit(X, y)
    return model


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def battery_metrics(y_true, y_pred, groups):
    frame = pd.DataFrame({
        "battery_id": groups,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    bat = frame.groupby("battery_id").agg(
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        pred_std=("y_pred", "std"),
    )
    metrics = regression_metrics(bat["y_true"].to_numpy(), bat["y_pred"].to_numpy())
    return metrics, bat


def stratified_group_holdout(df, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=SEED):
    """Split holdout por bateria, estratificado por faixa de SOH."""
    battery_table = (
        df.groupby("battery_id", as_index=False)[TARGET]
        .first()
        .rename(columns={TARGET: "target_soh"})
    )
    n_bins = min(10, max(2, len(battery_table) // 12))
    battery_table["bin"] = pd.qcut(
        battery_table["target_soh"], q=n_bins, labels=False, duplicates="drop"
    )

    rng = np.random.default_rng(seed)
    split_map = {}

    for _, group in battery_table.groupby("bin"):
        ids = group["battery_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round(n * val_size)))

        while n_test + n_val >= n and (n_test > 1 or n_val > 1):
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1

        for i, battery_id in enumerate(ids):
            if i < n_test:
                split_map[battery_id] = "test"
            elif i < n_test + n_val:
                split_map[battery_id] = "val"
            else:
                split_map[battery_id] = "train"

    split_df = pd.DataFrame({
        "battery_id": list(split_map.keys()),
        "split": list(split_map.values()),
    })
    return split_df


def apply_split(df, split_df):
    out = df.merge(split_df, on="battery_id", how="left", validate="many_to_one")
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "battery_id"].unique()[:5]
        raise ValueError(f"Baterias sem split: {missing}")
    return out


def cross_validate_grouped(df_train, feature_cols, models, seed=SEED):
    X = df_train[feature_cols].to_numpy()
    y = df_train[TARGET].to_numpy()
    groups = df_train["battery_id"].to_numpy()
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    rows = []
    for model_name, factory in models.items():
        for repeat in range(CV_REPEATS):
            splitter = StratifiedGroupKFold(
                n_splits=CV_FOLDS,
                shuffle=True,
                random_state=seed + repeat,
            )
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(X, y_bins, groups)
            ):
                model = clone(factory)
                start = time.perf_counter()
                fit_model(
                    model, model_name,
                    X[train_idx], y[train_idx], groups[train_idx],
                )
                fit_seconds = time.perf_counter() - start

                pred = np.clip(model.predict(X[test_idx]), 0.0, 100.0)
                row_metrics = regression_metrics(y[test_idx], pred)
                bat_metrics, _ = battery_metrics(
                    y[test_idx], pred, groups[test_idx]
                )

                rows.append({
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "row_mae": row_metrics["mae"],
                    "row_rmse": row_metrics["rmse"],
                    "row_r2": row_metrics["r2"],
                    "battery_mae": bat_metrics["mae"],
                    "battery_rmse": bat_metrics["rmse"],
                    "battery_r2": bat_metrics["r2"],
                    "fit_seconds": fit_seconds,
                })

    folds = pd.DataFrame(rows)
    summary = folds.groupby("model", as_index=False).agg(
        cv_mae_mean=("row_mae", "mean"),
        cv_mae_std=("row_mae", "std"),
        cv_rmse_mean=("row_rmse", "mean"),
        cv_rmse_std=("row_rmse", "std"),
        cv_r2_mean=("row_r2", "mean"),
        cv_r2_std=("row_r2", "std"),
        cv_battery_mae_mean=("battery_mae", "mean"),
        cv_battery_mae_std=("battery_mae", "std"),
        cv_battery_rmse_mean=("battery_rmse", "mean"),
        cv_battery_rmse_std=("battery_rmse", "std"),
        cv_battery_r2_mean=("battery_r2", "mean"),
        cv_battery_r2_std=("battery_r2", "std"),
        cv_fit_seconds_mean=("fit_seconds", "mean"),
    )
    summary = summary.sort_values(["cv_mae_mean", "cv_rmse_mean"]).reset_index(drop=True)
    return summary, folds


def evaluate_models_on_set(models, df_fit, df_eval, feature_cols):
    X_fit = df_fit[feature_cols].to_numpy()
    y_fit = df_fit[TARGET].to_numpy()
    g_fit = df_fit["battery_id"].to_numpy()

    X_eval = df_eval[feature_cols].to_numpy()
    y_eval = df_eval[TARGET].to_numpy()
    g_eval = df_eval["battery_id"].to_numpy()

    rows = []
    for model_name, factory in models.items():
        model = clone(factory)
        start = time.perf_counter()
        fit_model(model, model_name, X_fit, y_fit, g_fit)
        fit_seconds = time.perf_counter() - start

        pred = np.clip(model.predict(X_eval), 0.0, 100.0)
        row_metrics = regression_metrics(y_eval, pred)
        bat_metrics, _ = battery_metrics(y_eval, pred, g_eval)

        rows.append({
            "model": model_name,
            "val_mae": row_metrics["mae"],
            "val_rmse": row_metrics["rmse"],
            "val_r2": row_metrics["r2"],
            "val_battery_mae": bat_metrics["mae"],
            "val_battery_rmse": bat_metrics["rmse"],
            "val_battery_r2": bat_metrics["r2"],
            "val_fit_seconds": fit_seconds,
        })

    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=SEED, statistic=np.mean):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = statistic(values[idx], axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def grouped_bootstrap_metrics(df_pred):
    """IC bootstrap reamostrando baterias, para métricas por linha e por bateria."""
    batteries = df_pred["battery_id"].unique()
    battery_table = df_pred.groupby("battery_id").agg(
        y_true=(TARGET, "first"),
        y_pred=("prediction", "mean"),
    ).loc[batteries]
    row_groups = {
        battery_id: df_pred[df_pred["battery_id"] == battery_id]
        for battery_id in batteries
    }

    rng = np.random.default_rng(SEED)
    row_mae, row_rmse, bat_mae, bat_rmse = [], [], [], []

    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(batteries, size=len(batteries), replace=True)

        boot = pd.concat(
            [row_groups[battery_id] for battery_id in sample],
            ignore_index=True,
        )
        y_true = boot[TARGET].to_numpy()
        y_pred = boot["prediction"].to_numpy()
        row_mae.append(mean_absolute_error(y_true, y_pred))
        row_rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))

        y_true_bat = battery_table.loc[sample, "y_true"].to_numpy()
        y_pred_bat = battery_table.loc[sample, "y_pred"].to_numpy()
        bat_mae.append(mean_absolute_error(y_true_bat, y_pred_bat))
        bat_rmse.append(np.sqrt(mean_squared_error(y_true_bat, y_pred_bat)))

    def ci(values):
        return (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )

    return {
        "row_mae_ci": ci(row_mae),
        "row_rmse_ci": ci(row_rmse),
        "battery_mae_ci": ci(bat_mae),
        "battery_rmse_ci": ci(bat_rmse),
    }


def save_feature_importance(model, model_name, X_val, y_val, feature_cols, path):
    try:
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
        importance = None
        if hasattr(estimator, "feature_importances_"):
            importance = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            importance = np.abs(np.ravel(estimator.coef_))

        if importance is None or len(importance) != len(feature_cols):
            result = permutation_importance(
                model, X_val, y_val,
                n_repeats=10,
                random_state=SEED,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            importance = np.asarray(result.importances_mean, dtype=float)

        pd.DataFrame({
            "feature": feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False).to_csv(path, index=False)
    except Exception as exc:
        print(f"  Não foi possível calcular importância para {model_name}: {exc}")


def residual_analysis(df_pred, dataset_name, output_dir):
    df_pred = df_pred.copy()
    df_pred["error"] = df_pred["prediction"] - df_pred[TARGET]
    df_pred["abs_error"] = df_pred["error"].abs()
    df_pred["ape"] = df_pred["abs_error"] / df_pred[TARGET] * 100.0

    df_pred.to_csv(
        os.path.join(output_dir, f"test_predictions_{dataset_name}.csv"),
        index=False,
    )

    soh_bands = pd.cut(
        df_pred[TARGET],
        bins=[0, 40, 50, 60, 70, 80, 90, 101],
        labels=["<40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"],
    ).rename("soh_band")
    by_soh = df_pred.groupby(soh_bands, observed=True).agg(
        n=("abs_error", "size"),
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
        mape=("ape", "mean"),
        p90_abs_error=("abs_error", lambda x: x.quantile(0.9)),
    ).reset_index()
    by_soh.to_csv(
        os.path.join(output_dir, f"error_by_soh_{dataset_name}.csv"),
        index=False,
    )

    df_pred["soc_band"] = pd.qcut(
        df_pred["soc_pct"], q=4, duplicates="drop"
    )
    by_soc = df_pred.groupby("soc_band", observed=True).agg(
        n=("abs_error", "size"),
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()
    by_soc.to_csv(
        os.path.join(output_dir, f"error_by_soc_{dataset_name}.csv"),
        index=False,
    )

    df_pred["temperature_band"] = pd.qcut(
        df_pred["temperature_c"], q=4, duplicates="drop"
    )
    by_temp = df_pred.groupby("temperature_band", observed=True).agg(
        n=("abs_error", "size"),
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
    ).reset_index()
    by_temp.to_csv(
        os.path.join(output_dir, f"error_by_temperature_{dataset_name}.csv"),
        index=False,
    )

    battery = df_pred.groupby("battery_id").agg(
        target_soh=(TARGET, "first"),
        prediction_mean=("prediction", "mean"),
        prediction_median=("prediction", "median"),
        prediction_std=("prediction", "std"),
    )
    battery["error_mean"] = battery["prediction_mean"] - battery["target_soh"]
    battery["abs_error_mean"] = battery["error_mean"].abs()
    battery.to_csv(
        os.path.join(output_dir, f"battery_predictions_{dataset_name}.csv")
    )

    return df_pred, battery


def plot_predictions(df_pred, battery, dataset_name, model_name, output_dir):
    plt.figure(figsize=(7, 6))
    plt.scatter(
        df_pred[TARGET], df_pred["prediction"],
        alpha=0.45, label="cenários",
    )
    plt.scatter(
        battery["target_soh"], battery["prediction_mean"],
        marker="x", s=45, label="média por bateria",
    )
    plt.plot([0, 100], [0, 100], linestyle="--", color="black")
    plt.xlabel("SOH real medido por C20 simulado (%)")
    plt.ylabel("SOH estimado (%)")
    plt.title(
        f"{dataset_name}: {model_name} | "
        f"MAE bateria = {battery['abs_error_mean'].mean():.2f} pp"
    )
    plt.xlim(20, 102)
    plt.ylim(20, 102)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"predictions_{dataset_name}.png"), dpi=160
    )
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(
        df_pred[TARGET], df_pred["error"],
        alpha=0.45, color="tab:red",
    )
    plt.axhline(0.0, linestyle="--", color="black")
    plt.xlabel("SOH real (%)")
    plt.ylabel("Erro (estimado - real, pp)")
    plt.title(f"{dataset_name}: resíduos do modelo {model_name}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"residuals_{dataset_name}.png"), dpi=160
    )
    plt.close()


def run_dataset(dataset_name, df, feature_cols, split_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    assert_no_leakage(feature_cols)
    df = apply_split(df, split_df)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    print(
        f"\n=== {dataset_name} ===\n"
        f"treino: {train.battery_id.nunique()} baterias / {len(train)} linhas | "
        f"val: {val.battery_id.nunique()} / {len(val)} | "
        f"teste: {test.battery_id.nunique()} / {len(test)}"
    )

    models = make_models(feature_cols)
    cv_summary, cv_folds = cross_validate_grouped(train, feature_cols, models)
    cv_summary.insert(0, "dataset", dataset_name)
    cv_summary.to_csv(
        os.path.join(output_dir, f"cv_results_{dataset_name}.csv"), index=False
    )
    cv_folds.insert(0, "dataset", dataset_name)
    cv_folds.to_csv(
        os.path.join(output_dir, f"cv_folds_{dataset_name}.csv"), index=False
    )

    val_results = evaluate_models_on_set(models, train, val, feature_cols)
    val_results.insert(0, "dataset", dataset_name)
    val_results.to_csv(
        os.path.join(output_dir, f"val_results_{dataset_name}.csv"), index=False
    )

    best_row = cv_summary.sort_values(
        ["cv_mae_mean", "cv_rmse_mean"]
    ).iloc[0]
    best_name = best_row["model"]
    print(f"Modelo selecionado por CV: {best_name}")

    final_model = clone(make_models(feature_cols)[best_name])
    train_val = pd.concat([train, val], ignore_index=True)
    fit_model(
        final_model, best_name,
        train_val[feature_cols].to_numpy(),
        train_val[TARGET].to_numpy(),
        train_val["battery_id"].to_numpy(),
    )

    X_test = test[feature_cols].to_numpy()
    y_test = test[TARGET].to_numpy()
    g_test = test["battery_id"].to_numpy()
    test_pred = np.clip(final_model.predict(X_test), 0.0, 100.0)

    test_df = test.copy()
    test_df["prediction"] = test_pred
    test_df = test_df[[
        "battery_id", "scenario_id", TARGET,
        "soc_pct", "temperature_c", "prediction",
    ]]

    row_metrics = regression_metrics(y_test, test_pred)
    bat_metrics, battery = battery_metrics(y_test, test_pred, g_test)
    cis = grouped_bootstrap_metrics(test_df)

    test_metrics = pd.DataFrame([
        {
            "dataset": dataset_name,
            "model": best_name,
            "level": "row",
            "mae": row_metrics["mae"],
            "mae_ci_low": cis["row_mae_ci"][0],
            "mae_ci_high": cis["row_mae_ci"][1],
            "rmse": row_metrics["rmse"],
            "rmse_ci_low": cis["row_rmse_ci"][0],
            "rmse_ci_high": cis["row_rmse_ci"][1],
            "r2": row_metrics["r2"],
        },
        {
            "dataset": dataset_name,
            "model": best_name,
            "level": "battery",
            "mae": bat_metrics["mae"],
            "mae_ci_low": cis["battery_mae_ci"][0],
            "mae_ci_high": cis["battery_mae_ci"][1],
            "rmse": bat_metrics["rmse"],
            "rmse_ci_low": cis["battery_rmse_ci"][0],
            "rmse_ci_high": cis["battery_rmse_ci"][1],
            "r2": bat_metrics["r2"],
        },
    ])
    test_metrics.to_csv(
        os.path.join(output_dir, f"test_metrics_{dataset_name}.csv"), index=False
    )

    test_df, battery = residual_analysis(test_df, dataset_name, output_dir)
    plot_predictions(test_df, battery, dataset_name, best_name, output_dir)

    # Importância calculada com um modelo ajustado apenas no treino, avaliado
    # na validação. Usar o modelo final (treino+val) aqui seria in-sample.
    importance_model = clone(make_models(feature_cols)[best_name])
    fit_model(
        importance_model, best_name,
        train[feature_cols].to_numpy(),
        train[TARGET].to_numpy(),
        train["battery_id"].to_numpy(),
    )
    save_feature_importance(
        importance_model, best_name,
        val[feature_cols].to_numpy(), val[TARGET].to_numpy(),
        feature_cols,
        os.path.join(output_dir, f"feature_importance_{dataset_name}.csv"),
    )

    joblib.dump(
        final_model,
        os.path.join(output_dir, f"{dataset_name}_selected_{best_name}.joblib"),
    )

    return {
        "dataset": dataset_name,
        "selected_model": best_name,
        "cv_mae_mean": best_row["cv_mae_mean"],
        "cv_mae_std": best_row["cv_mae_std"],
        "cv_rmse_mean": best_row["cv_rmse_mean"],
        "cv_battery_mae_mean": best_row["cv_battery_mae_mean"],
        "test_mae_row": row_metrics["mae"],
        "test_rmse_row": row_metrics["rmse"],
        "test_r2_row": row_metrics["r2"],
        "test_mae_battery": bat_metrics["mae"],
        "test_rmse_battery": bat_metrics["rmse"],
        "test_r2_battery": bat_metrics["r2"],
        "test_mae_battery_ci_low": cis["battery_mae_ci"][0],
        "test_mae_battery_ci_high": cis["battery_mae_ci"][1],
        "n_train_batteries": train.battery_id.nunique(),
        "n_val_batteries": val.battery_id.nunique(),
        "n_test_batteries": test.battery_id.nunique(),
    }, cv_summary, val_results, test_metrics


def main(pulse_path, crank_path, combined_path, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    pulse = pd.read_csv(pulse_path)
    crank = pd.read_csv(crank_path)
    combined = pd.read_csv(combined_path)

    if "capacity_ah_simulated" in pulse.columns:
        raise ValueError(
            "capacity_ah_simulated encontrada no dataset de features; "
            "remova-a para evitar vazamento."
        )

    split_df = stratified_group_holdout(pulse)
    split_df.to_csv(os.path.join(output_dir, "split_manifest.csv"), index=False)

    datasets = {
        "pulse": (pulse, PULSE_FEATURES),
        "cranking": (crank, CRANK_FEATURES),
        "combined": (combined, COMBINED_FEATURES),
    }

    selected_rows = []
    cv_tables = []
    val_tables = []
    test_tables = []

    for dataset_name, (df, feature_cols) in datasets.items():
        row, cv, val, testmet = run_dataset(
            dataset_name, df, feature_cols, split_df, output_dir
        )
        selected_rows.append(row)
        cv_tables.append(cv)
        val_tables.append(val)
        test_tables.append(testmet)

    comparison = pd.concat(cv_tables, ignore_index=True).merge(
        pd.concat(val_tables, ignore_index=True),
        on=["dataset", "model"],
        how="left",
    )

    selected_df = pd.DataFrame(selected_rows)
    selected_map = selected_df.set_index("dataset")["selected_model"].to_dict()
    comparison["selected_model"] = comparison["dataset"].map(selected_map)

    test_all = pd.concat(test_tables, ignore_index=True)
    test_selected = test_all[test_all["level"] == "battery"][[
        "dataset", "model", "mae", "rmse", "r2",
        "mae_ci_low", "mae_ci_high",
    ]].rename(columns={
        "model": "selected_model",
        "mae": "test_mae_battery",
        "rmse": "test_rmse_battery",
        "r2": "test_r2_battery",
        "mae_ci_low": "test_mae_battery_ci_low",
        "mae_ci_high": "test_mae_battery_ci_high",
    })

    comparison = comparison.merge(
        test_selected,
        on=["dataset", "selected_model"],
        how="left",
    )
    comparison.to_csv(
        os.path.join(output_dir, "comparison_all.csv"), index=False
    )
    selected_df.to_csv(
        os.path.join(output_dir, "selected_models.csv"), index=False
    )

    print("\n=== MODELOS SELECIONADOS ===")
    print(selected_df.to_string(index=False))
    print("\n=== COMPARAÇÃO (CV + validação; teste apenas do selecionado) ===")
    print(
        comparison.sort_values(["dataset", "cv_mae_mean"]).to_string(index=False)
    )

    return selected_df, comparison


if __name__ == "__main__":
    main(
        "outputs/pulse_dataset.csv",
        "outputs/cranking_dataset.csv",
        "outputs/combined_dataset.csv",
    )
