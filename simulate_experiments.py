"""Geração de características de pulso, cranking e dataset combinado.

Mudanças principais em relação à versão inicial:
- cenários SOC/temperatura por Latin Hypercube, em vez de 8 de 16 combinações;
- relaxamento pós-pulso simulado a partir do estado de polarização real;
- features de cranking baseadas no mínimo de tensão e na recuperação, sem os
  dois mínimos locais adjacentes que eram apenas ruído;
- capacidade de referência medida por descarga C20 simulada;
- erros leves de ganho de tensão/corrente e offset de temperatura;
- a coluna de capacidade não é incluída nos CSVs de features.
"""
import numpy as np
import pandas as pd

from battery_model import (
    make_battery,
    measure_c20_capacity,
    ocv_from_soc,
    sample_soh,
    simulate_ecm,
)
from config import (
    CRANK_DURATION_S,
    CRANK_SAMPLE_RATE_HZ,
    C20_TEMPERATURE_C,
    CURRENT_NOISE_STD_A,
    N_BATTERIES,
    N_SCENARIOS,
    NOMINAL_CAPACITY_AH,
    PULSE_CURRENT_A,
    PULSE_DURATION_S,
    PULSE_SAMPLE_RATE_HZ,
    REST_DURATION_S,
    REST_SAMPLE_RATE_HZ,
    SOC_NOISE_STD_PCT,
    SOC_RANGE,
    TEMP_RANGE,
    TEMPERATURE_NOISE_STD_C,
    VOLTAGE_NOISE_STD_V,
)


def latin_hypercube(n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube 2D simples, sem dependência externa."""
    u = np.empty((n, 2), dtype=float)
    for j in range(2):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def _tau_from_recovery(recovery_v: np.ndarray, v_start: float, v_final: float, fs: int) -> float:
    """Tempo aproximado para 63% da recuperação total."""
    gain = v_final - v_start
    if gain <= 1e-4 or len(recovery_v) < 2:
        return float(REST_DURATION_S)

    window = max(1, int(0.05 * fs))
    kernel = np.ones(window) / window
    pad = window // 2
    smooth = np.convolve(
        np.pad(recovery_v, (pad, pad), mode="edge"),
        kernel,
        mode="valid",
    )
    target = v_start + 0.632 * gain
    idx = int(np.argmax(smooth >= target))
    if smooth[idx] < target:
        return float(REST_DURATION_S)
    return float(idx / fs)


def simulate_pulse(battery, soc_true, temp_true, rng):
    fs = PULSE_SAMPLE_RATE_HZ
    dt = 1.0 / fs
    n = int(PULSE_DURATION_S * fs)

    ocv_true = ocv_from_soc(soc_true, temp_true)
    ocv_measured = (
        ocv_true * battery.voltage_gain
        + rng.normal(0.0, VOLTAGE_NOISE_STD_V)
    )

    current = np.full(n, PULSE_CURRENT_A, dtype=float)
    voltage, current_measured, vp1, vp2 = simulate_ecm(
        battery, current, soc_true, temp_true, dt, rng,
        VOLTAGE_NOISE_STD_V, CURRENT_NOISE_STD_A,
    )

    window = max(1, int(0.1 * fs))
    v_loaded = float(np.mean(voltage[-window:]))
    i_mean = float(np.mean(current_measured))
    delta_v = ocv_measured - v_loaded
    r_dynamic_mohm = delta_v / max(abs(i_mean), 1e-9) * 1000.0

    v_first = float(voltage[0])
    r_early_mohm = (
        (ocv_measured - v_first)
        / max(abs(current_measured[0]), 1e-9)
        * 1000.0
    )

    # Relaxamento a partir do estado de polarização real do fim do pulso.
    # A taxa de amostragem do repouso pode ser menor porque as constantes de
    # tempo relevantes são de segundos.
    rest_fs = REST_SAMPLE_RATE_HZ
    rest_dt = 1.0 / rest_fs
    rest_n = int(REST_DURATION_S * rest_fs)
    rest_window = max(1, int(0.1 * rest_fs))
    rest_current = np.zeros(rest_n, dtype=float)
    rest_voltage, _, _, _ = simulate_ecm(
        battery, rest_current, soc_true, temp_true, rest_dt, rng,
        VOLTAGE_NOISE_STD_V, CURRENT_NOISE_STD_A,
        initial_vp1=vp1, initial_vp2=vp2,
    )
    v_recovery = float(np.mean(rest_voltage[-rest_window:]))
    recovery_gain = v_recovery - v_loaded
    relaxation_ratio = recovery_gain / max(delta_v, 1e-9)
    # O primeiro instante do repouso já inclui a recuperação ôhmica imediata;
    # a constante de tempo deve ser medida a partir daí, não de v_loaded.
    v_rest_start = float(
        np.mean(rest_voltage[:max(1, int(0.005 * rest_fs))])
    )
    tau_recovery_s = _tau_from_recovery(
        rest_voltage, v_rest_start, v_recovery, rest_fs
    )

    return {
        "pulse_ocv_v": float(ocv_measured),
        "pulse_v_loaded_v": v_loaded,
        "pulse_delta_v": float(delta_v),
        "pulse_current_a": i_mean,
        "pulse_r_dynamic_mohm": float(r_dynamic_mohm),
        "pulse_r_early_mohm": float(r_early_mohm),
        "pulse_v_recovery_v": v_recovery,
        "pulse_recovery_delta_v": float(recovery_gain),
        "pulse_relaxation_ratio": float(np.clip(relaxation_ratio, -1.0, 1.5)),
        "pulse_tau_recovery_s": tau_recovery_s,
    }


def make_crank_current(n, fs, rng):
    """Perfil sintético ilustrativo de corrente de partida."""
    t = np.arange(n) / fs
    current = np.zeros(n)

    active = (t >= 0.15) & (t <= 2.20)
    tau = t[active] - 0.15
    envelope = 1.0 - 0.15 * np.clip(tau / 2.05, 0, 1)
    pulsation = 1.0 + 0.12 * np.sin(2 * np.pi * 8.0 * tau)
    current[active] = 220.0 * envelope * pulsation

    current *= rng.normal(1.0, 0.08)
    current += rng.normal(0.0, 1.0, n)
    return np.maximum(current, 0.0)


def simulate_cranking(battery, soc_true, temp_true, rng):
    fs = CRANK_SAMPLE_RATE_HZ
    dt = 1.0 / fs
    n = int(CRANK_DURATION_S * fs)

    current = make_crank_current(n, fs, rng)
    ocv_true = ocv_from_soc(soc_true, temp_true)
    ocv_measured = (
        ocv_true * battery.voltage_gain
        + rng.normal(0.0, VOLTAGE_NOISE_STD_V)
    )

    voltage, current_measured, vp1, vp2 = simulate_ecm(
        battery, current, soc_true, temp_true, dt, rng,
        VOLTAGE_NOISE_STD_V, CURRENT_NOISE_STD_A,
    )

    active_start = int(0.15 * fs)
    active_end = int(2.20 * fs)
    active_v = voltage[active_start:active_end]
    active_i = current_measured[active_start:active_end]

    idx_min = active_start + int(np.argmin(active_v))
    v_min = float(voltage[idx_min])
    i_at_min = float(current_measured[idx_min])
    i_mean = float(np.mean(active_i))
    i_max = float(np.max(active_i))
    v_mean = float(np.mean(active_v))
    delta_v_min = ocv_measured - v_min
    r_eff_mohm = delta_v_min / max(abs(i_at_min), 1e-9) * 1000.0

    window = max(1, int(0.1 * fs))
    v_end = float(np.mean(voltage[-window:]))

    rest_fs = REST_SAMPLE_RATE_HZ
    rest_dt = 1.0 / rest_fs
    rest_n = int(REST_DURATION_S * rest_fs)
    rest_window = max(1, int(0.1 * rest_fs))
    rest_current = np.zeros(rest_n, dtype=float)
    rest_voltage, _, _, _ = simulate_ecm(
        battery, rest_current, soc_true, temp_true, rest_dt, rng,
        VOLTAGE_NOISE_STD_V, CURRENT_NOISE_STD_A,
        initial_vp1=vp1, initial_vp2=vp2,
    )
    v_recovery = float(np.mean(rest_voltage[-rest_window:]))
    recovery_gain = v_recovery - v_end
    tau_recovery_s = _tau_from_recovery(
        rest_voltage, v_end, v_recovery, rest_fs
    )

    return {
        "crank_ocv_v": float(ocv_measured),
        "crank_v_min_v": v_min,
        "crank_v_mean_v": v_mean,
        "crank_delta_v_min_v": float(delta_v_min),
        "crank_i_max_a": i_max,
        "crank_i_mean_a": i_mean,
        "crank_r_eff_mohm": float(r_eff_mohm),
        "crank_t_v_min_s": float(idx_min / fs),
        "crank_v_end_v": v_end,
        "crank_v_recovery_v": v_recovery,
        "crank_recovery_gain_v": float(recovery_gain),
        "crank_tau_recovery_s": tau_recovery_s,
    }


def generate_datasets(seed=42, n_batteries=N_BATTERIES):
    rng = np.random.default_rng(seed)
    pulse_rows = []
    crank_rows = []
    metadata_rows = []

    soh_parameters = sample_soh(rng, n_batteries)

    for b_idx, soh_param in enumerate(soh_parameters):
        battery_id = f"BAT{b_idx + 1:04d}"
        battery = make_battery(rng, battery_id, soh_param)

        capacity_measured = measure_c20_capacity(
            battery, C20_TEMPERATURE_C, rng
        )
        target_soh = capacity_measured / NOMINAL_CAPACITY_AH * 100.0

        metadata_rows.append({
            "battery_id": battery_id,
            "target_soh": float(target_soh),
            "capacity_ah_measured": float(capacity_measured),
            "capacity_ah_parameter": float(battery.capacity_ah),
            "soh_capacity_parameter": float(battery.soh_capacity),
            "resistance_health": float(battery.resistance_health),
            "r0_ohm": float(battery.r0_ohm),
            "r1_ohm": float(battery.r1_ohm),
            "r2_ohm": float(battery.r2_ohm),
            "c1_f": float(battery.c1_f),
            "c2_f": float(battery.c2_f),
            "voltage_gain": float(battery.voltage_gain),
            "current_gain": float(battery.current_gain),
            "temperature_offset_c": float(battery.temperature_offset_c),
        })

        u = latin_hypercube(N_SCENARIOS, rng)
        soc_true_values = SOC_RANGE[0] + u[:, 0] * (SOC_RANGE[1] - SOC_RANGE[0])
        temp_true_values = TEMP_RANGE[0] + u[:, 1] * (TEMP_RANGE[1] - TEMP_RANGE[0])

        for scenario_idx, (soc_true, temp_true) in enumerate(
            zip(soc_true_values, temp_true_values)
        ):
            soc_measured = float(
                np.clip(
                    soc_true + rng.normal(0.0, SOC_NOISE_STD_PCT),
                    0.0,
                    100.0,
                )
            )
            temp_measured = float(
                temp_true
                + battery.temperature_offset_c
                + rng.normal(0.0, TEMPERATURE_NOISE_STD_C)
            )

            common = {
                "battery_id": battery_id,
                "scenario_id": f"{battery_id}_S{scenario_idx + 1:02d}",
                "target_soh": float(target_soh),
                "soc_pct": soc_measured,
                "temperature_c": temp_measured,
            }

            pulse = simulate_pulse(battery, soc_true, temp_true, rng)
            crank = simulate_cranking(battery, soc_true, temp_true, rng)

            pulse_rows.append({**common, **pulse})
            crank_rows.append({**common, **crank})

    pulse_df = pd.DataFrame(pulse_rows)
    crank_df = pd.DataFrame(crank_rows)
    metadata_df = pd.DataFrame(metadata_rows)

    combined_df = pulse_df.merge(
        crank_df,
        on=[
            "battery_id", "scenario_id", "target_soh",
            "soc_pct", "temperature_c",
        ],
        how="inner",
    )

    return pulse_df, crank_df, combined_df, metadata_df
