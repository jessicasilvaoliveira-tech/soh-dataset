"""Modelo equivalente Thevenin 2RC simplificado para bateria chumbo-ácido.

Nesta versão:
- a capacidade e a resistência têm componentes de envelhecimento parcialmente
  independentes;
- a capacidade de referência é medida por um ensaio C20 simulado, e não apenas
  atribuída;
- o relaxamento pós-pulso é simulado a partir do estado de polarização real;
- erros de ganho/offset de sensores são incluídos de forma leve.
"""
from dataclasses import dataclass

import numpy as np

from config import (
    CAPACITY_FACTOR_STD,
    C20_CURRENT_A,
    C20_CUTOFF_V,
    C20_DT_S,
    CURRENT_GAIN_STD,
    CURRENT_NOISE_STD_A,
    NOMINAL_CAPACITY_AH,
    RESISTANCE_FACTOR_SIGMA,
    RESISTANCE_HEALTH_STD,
    SOH_HIGH_BETA_A,
    SOH_HIGH_BETA_B,
    SOH_LOW_BETA_A,
    SOH_LOW_BETA_B,
    SOH_LOW_FRACTION,
    SOH_MAX,
    SOH_MIN,
    TEMPERATURE_OFFSET_STD_C,
    VOLTAGE_GAIN_STD,
    VOLTAGE_NOISE_STD_V,
)


@dataclass
class Battery:
    battery_id: str
    soh_capacity: float
    capacity_ah: float
    resistance_health: float
    r0_ohm: float
    r1_ohm: float
    c1_f: float
    r2_ohm: float
    c2_f: float
    capacity_factor: float
    resistance_factor: float
    voltage_gain: float
    current_gain: float
    temperature_offset_c: float


def sample_soh(rng: np.random.Generator, n: int) -> np.ndarray:
    """Amostra SOH de uma mistura com concentração em SOH alto e cauda baixa."""
    is_low = rng.random(n) < SOH_LOW_FRACTION
    high = SOH_MIN + (SOH_MAX - SOH_MIN) * rng.beta(
        SOH_HIGH_BETA_A, SOH_HIGH_BETA_B, size=n
    )
    low = SOH_MIN + (SOH_MAX - SOH_MIN) * rng.beta(
        SOH_LOW_BETA_A, SOH_LOW_BETA_B, size=n
    )
    return np.where(is_low, low, high)


def ocv_from_soc(soc_pct: float, temperature_c: float) -> float:
    """Curva OCV-SOC ilustrativa, com cauda de descarga em SOC muito baixo.

    A região 10--100% é aproximadamente linear (11.89--12.70 V). Abaixo de 10%
    a curva cai até 10.60 V, permitindo representar o corte de um ensaio C20
    sem introduzir uma queda artificial de resistência.
    """
    soc = float(np.clip(soc_pct / 100.0, 0.0, 1.0))
    if soc >= 0.10:
        ocv = 11.80 + 0.90 * soc
    else:
        ocv = 11.89 - (11.89 - 10.45) * (0.10 - soc) / 0.10
    ocv += 0.0008 * (temperature_c - 25.0)
    return float(ocv)


def make_battery(rng: np.random.Generator, battery_id: str, soh_parameter: float) -> Battery:
    """Cria parâmetros individuais com capacidade e resistência desacopladas."""
    capacity_factor = float(rng.normal(1.0, CAPACITY_FACTOR_STD))
    actual_capacity_soh = float(
        np.clip(soh_parameter * capacity_factor, SOH_MIN, SOH_MAX)
    )
    capacity_ah = NOMINAL_CAPACITY_AH * actual_capacity_soh / 100.0

    # A saúde resistiva acompanha a capacidade, mas com um desvio próprio.
    resistance_health = float(
        np.clip(
            actual_capacity_soh + rng.normal(0.0, RESISTANCE_HEALTH_STD),
            SOH_MIN,
            SOH_MAX,
        )
    )

    resistance_factor = float(
        rng.lognormal(mean=0.0, sigma=RESISTANCE_FACTOR_SIGMA)
    )
    degradation = max(0.0, 1.0 - resistance_health / 100.0)

    r0_new = rng.uniform(0.0042, 0.0055)
    r0 = r0_new * (1.0 + 2.0 * degradation**1.35) * resistance_factor
    r1 = rng.uniform(0.0012, 0.0019) * (1.0 + 1.2 * degradation) * resistance_factor
    r2 = rng.uniform(0.0018, 0.0030) * (1.0 + 1.8 * degradation) * resistance_factor
    c1 = rng.uniform(2200.0, 3400.0) * (1.0 - 0.20 * degradation)
    c2 = rng.uniform(9000.0, 13000.0) * (1.0 - 0.25 * degradation)

    voltage_gain = float(np.clip(rng.normal(1.0, VOLTAGE_GAIN_STD), 0.95, 1.05))
    current_gain = float(np.clip(rng.normal(1.0, CURRENT_GAIN_STD), 0.90, 1.10))
    temperature_offset = float(rng.normal(0.0, TEMPERATURE_OFFSET_STD_C))

    return Battery(
        battery_id=battery_id,
        soh_capacity=actual_capacity_soh,
        capacity_ah=float(capacity_ah),
        resistance_health=resistance_health,
        r0_ohm=float(r0),
        r1_ohm=float(r1),
        c1_f=float(c1),
        r2_ohm=float(r2),
        c2_f=float(c2),
        capacity_factor=float(capacity_factor),
        resistance_factor=float(resistance_factor),
        voltage_gain=voltage_gain,
        current_gain=current_gain,
        temperature_offset_c=temperature_offset,
    )


def temperature_soc_adjustments(temperature_c: float, soc_pct: float):
    """Fatores exploratórios: resistência cresce no frio e em SOC baixo."""
    cold_factor = np.exp(0.018 * (25.0 - temperature_c))
    low_soc_factor = (
        1.0
        + 0.45 * ((100.0 - soc_pct) / 100.0)
        + 0.80 * max(0.0, (20.0 - soc_pct) / 20.0) ** 2
    )
    return float(cold_factor), float(low_soc_factor)


def simulate_ecm(
    battery: Battery,
    current_a: np.ndarray,
    soc_pct: float,
    temperature_c: float,
    dt: float,
    rng: np.random.Generator,
    voltage_noise_std: float = VOLTAGE_NOISE_STD_V,
    current_noise_std: float = CURRENT_NOISE_STD_A,
    initial_vp1: float = 0.0,
    initial_vp2: float = 0.0,
):
    """Simula tensão terminal usando ECM 2RC discreto.

    Convenção: corrente positiva = descarga. Retorna também o estado final das
    polarizações, para que o relaxamento possa ser simulado corretamente.
    """
    current_true = np.asarray(current_a, dtype=float)
    n = len(current_true)
    voltage = np.zeros(n)

    current_measured = (
        current_true * battery.current_gain
        + rng.normal(0.0, current_noise_std, n)
    )

    cold_factor, low_soc_factor = temperature_soc_adjustments(
        temperature_c, soc_pct
    )
    r0 = battery.r0_ohm * cold_factor * low_soc_factor
    polarization_soc_factor = 1.0 + 0.25 * (low_soc_factor - 1.0)
    r1 = battery.r1_ohm * cold_factor * polarization_soc_factor
    r2 = battery.r2_ohm * cold_factor * polarization_soc_factor

    vp1 = float(initial_vp1)
    vp2 = float(initial_vp2)
    ocv = ocv_from_soc(soc_pct, temperature_c)

    a1 = np.exp(-dt / max(r1 * battery.c1_f, 1e-9))
    a2 = np.exp(-dt / max(r2 * battery.c2_f, 1e-9))

    for k, current in enumerate(current_true):
        vp1 = a1 * vp1 + r1 * (1.0 - a1) * current
        vp2 = a2 * vp2 + r2 * (1.0 - a2) * current
        v_true = ocv - current * r0 - vp1 - vp2
        voltage[k] = v_true * battery.voltage_gain + rng.normal(
            0.0, voltage_noise_std
        )

    return voltage, current_measured, vp1, vp2


def measure_c20_capacity(
    battery: Battery,
    temperature_c: float,
    rng: np.random.Generator,
    dt: float = C20_DT_S,
    cutoff_v: float = C20_CUTOFF_V,
    current_a: float = C20_CURRENT_A,
) -> float:
    """Mede a capacidade por descarga C20 simulada até o corte de tensão.

    O resultado é a integral de corrente (Ah) até a tensão terminal atingir o
    corte. A capacidade não é simplesmente lida do parâmetro do modelo: ela
    depende da resistência, da temperatura e da curva OCV.
    """
    soc = 100.0
    vp1 = 0.0
    vp2 = 0.0
    ah = 0.0
    max_steps = int(1.25 * 20.0 * 3600.0 / dt)

    for _ in range(max_steps):
        cold_factor, low_soc_factor = temperature_soc_adjustments(
            temperature_c, soc
        )
        r0 = battery.r0_ohm * cold_factor * low_soc_factor
        polarization_soc_factor = 1.0 + 0.25 * (low_soc_factor - 1.0)
        r1 = battery.r1_ohm * cold_factor * polarization_soc_factor
        r2 = battery.r2_ohm * cold_factor * polarization_soc_factor

        a1 = np.exp(-dt / max(r1 * battery.c1_f, 1e-9))
        a2 = np.exp(-dt / max(r2 * battery.c2_f, 1e-9))
        vp1 = a1 * vp1 + r1 * (1.0 - a1) * current_a
        vp2 = a2 * vp2 + r2 * (1.0 - a2) * current_a

        ocv = ocv_from_soc(soc, temperature_c)
        v_terminal = ocv - current_a * r0 - vp1 - vp2
        v_measured = v_terminal + rng.normal(0.0, 0.001)

        ah += current_a * dt / 3600.0
        soc -= current_a * dt / (3600.0 * battery.capacity_ah) * 100.0

        if v_measured <= cutoff_v or soc <= 0.0:
            break

    return float(ah)
