"""Ponto de entrada: gera dados, salva artefatos e treina/avalia os modelos."""
import os

import numpy as np
import pandas as pd

from config import N_BATTERIES, OUTPUT_DIR, SEED
from simulate_experiments import generate_datasets
from train_evaluate import (
    COMBINED_FEATURES,
    CRANK_FEATURES,
    PULSE_FEATURES,
    main as train_main,
)


def build_dataset_summary(pulse, crank, combined, metadata):
    rows = []
    for name, df in [
        ("pulse", pulse),
        ("cranking", crank),
        ("combined", combined),
    ]:
        target = df["target_soh"]
        rows.append({
            "dataset": name,
            "n_rows": len(df),
            "n_batteries": df["battery_id"].nunique(),
            "n_scenarios_per_battery": int(
                df.groupby("battery_id").size().median()
            ),
            "n_missing": int(df.isna().sum().sum()),
            "n_duplicate_scenario_id": int(df.duplicated("scenario_id").sum()),
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "target_min": float(target.min()),
            "target_median": float(target.median()),
            "target_max": float(target.max()),
        })
    return pd.DataFrame(rows)


def build_feature_summary(pulse, crank, combined):
    rows = []
    for name, df, features in [
        ("pulse", pulse, PULSE_FEATURES),
        ("cranking", crank, CRANK_FEATURES),
        ("combined", combined, COMBINED_FEATURES),
    ]:
        for feature in features:
            series = df[feature]
            rows.append({
                "dataset": name,
                "feature": feature,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "corr_with_target": float(series.corr(df["target_soh"])),
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pulse, crank, combined, metadata = generate_datasets(
        seed=SEED,
        n_batteries=N_BATTERIES,
    )

    forbidden = {
        "capacity_ah_simulated",
        "capacity_ah_measured",
        "capacity_ah_parameter",
    }
    for name, df in [
        ("pulse", pulse),
        ("cranking", crank),
        ("combined", combined),
    ]:
        leaked = sorted(set(df.columns) & forbidden)
        if leaked:
            raise ValueError(f"{name} contém colunas de capacidade: {leaked}")

    pulse.to_csv(f"{OUTPUT_DIR}/pulse_dataset.csv", index=False)
    crank.to_csv(f"{OUTPUT_DIR}/cranking_dataset.csv", index=False)
    combined.to_csv(f"{OUTPUT_DIR}/combined_dataset.csv", index=False)
    metadata.to_csv(f"{OUTPUT_DIR}/battery_metadata.csv", index=False)

    summary = build_dataset_summary(pulse, crank, combined, metadata)
    summary.to_csv(f"{OUTPUT_DIR}/dataset_summary.csv", index=False)

    feature_summary = build_feature_summary(pulse, crank, combined)
    feature_summary.to_csv(f"{OUTPUT_DIR}/feature_summary.csv", index=False)

    soh_distribution = (
        metadata["target_soh"]
        .describe()
        .rename_axis("statistic")
        .reset_index()
        .rename(columns={"target_soh": "value"})
    )
    soh_distribution.to_csv(
        f"{OUTPUT_DIR}/soh_distribution.csv", index=False
    )

    print("Datasets gerados:")
    print(f"Pulso: {pulse.shape}")
    print(f"Cranking: {crank.shape}")
    print(f"Combinado: {combined.shape}")
    print(f"Metadados de baterias: {metadata.shape}")
    print("\nDistribuição do SOH medido por C20 simulado:")
    print(metadata["target_soh"].describe().round(2).to_string())

    train_main(
        f"{OUTPUT_DIR}/pulse_dataset.csv",
        f"{OUTPUT_DIR}/cranking_dataset.csv",
        f"{OUTPUT_DIR}/combined_dataset.csv",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
