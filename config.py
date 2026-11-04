"""Configurações do experimento sintético.

Os parâmetros físicos continuam sendo hipóteses exploratórias. O objetivo desta
versão é remover fontes óbvias de facilitação artificial, separar parcialmente
os mecanismos de perda de capacidade e aumento de resistência e adotar um
protocolo de avaliação honesto (validação agrupada + teste intocado).
"""
from pathlib import Path

SEED = 42

# ---------------------------------------------------------------------------
# Baterias e distribuição de SOH
# ---------------------------------------------------------------------------
N_BATTERIES = 320

# A distribuição deixa de ser uniforme. A maior parte da frota fica em SOH
# alto e uma fração menor representa baterias degradadas, como em uso real.
SOH_MIN = 30.0
SOH_MAX = 100.0
SOH_HIGH_BETA_A = 5.0
SOH_HIGH_BETA_B = 2.0
SOH_LOW_FRACTION = 0.15
SOH_LOW_BETA_A = 2.0
SOH_LOW_BETA_B = 5.0

# A capacidade e a resistência não são mais uma função unívoca uma da outra.
# resistance_health = capacity_soh + ruído normal com este desvio (em pp).
RESISTANCE_HEALTH_STD = 8.0
CAPACITY_FACTOR_STD = 0.03
RESISTANCE_FACTOR_SIGMA = 0.06

NOMINAL_CAPACITY_AH = 60.0
NOMINAL_VOLTAGE = 12.0
C20_CURRENT_A = NOMINAL_CAPACITY_AH / 20.0
C20_CUTOFF_V = 10.5
C20_DT_S = 5.0
C20_TEMPERATURE_C = 25.0

# ---------------------------------------------------------------------------
# Planejamento dos cenários
# ---------------------------------------------------------------------------
N_SCENARIOS = 8
SOC_RANGE = (40.0, 100.0)
TEMP_RANGE = (0.0, 45.0)
SOC_NOISE_STD_PCT = 1.5
TEMPERATURE_NOISE_STD_C = 0.3

# ---------------------------------------------------------------------------
# Pulso de corrente e cranking
# ---------------------------------------------------------------------------
PULSE_CURRENT_A = 20.0
PULSE_DURATION_S = 20.0
PULSE_SAMPLE_RATE_HZ = 200
REST_DURATION_S = 20.0
REST_SAMPLE_RATE_HZ = 100

CRANK_DURATION_S = 2.3
CRANK_SAMPLE_RATE_HZ = 1000

# ---------------------------------------------------------------------------
# Ruído e erros de medição
# ---------------------------------------------------------------------------
VOLTAGE_NOISE_STD_V = 0.003
CURRENT_NOISE_STD_A = 0.05
VOLTAGE_GAIN_STD = 0.0015
CURRENT_GAIN_STD = 0.005
TEMPERATURE_OFFSET_STD_C = 0.2

# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5
CV_REPEATS = 2
N_BOOTSTRAP = 2000

OUTPUT_DIR = "outputs"
