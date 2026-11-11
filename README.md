# Lead-Acid 12 V / 60 Ah

## Objetivo

Comparar modelos de regressão para estimar o SOH definido pela capacidade residual:

`SOH (%) = capacidade C20 medida (Ah) / 60 Ah * 100`

Os dados são inteiramente sintéticos. O ECM 2RC e as relações de envelhecimento são hipóteses exploratórias e precisam ser calibrados com medições reais antes de qualquer conclusão sobre baterias físicas.

## Conteúdo

- `config.py`: parâmetros do simulador, do planejamento experimental e da avaliação.
- `battery_model.py`: bateria virtual, ECM Thevenin 2RC, envelhecimento e ensaio C20 simulado.
- `simulate_experiments.py`: geração dos cenários de pulso e cranking e extração das features.
- `train_evaluate.py`: baselines, modelos de ML, validação agrupada, teste único, bootstrap e análise de resíduos.
- `main.py`: execução completa do pipeline.

## Instalação

Recomendado Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

## Protocolo experimental

1. São geradas 320 baterias com SOH de capacidade amostrado de uma mistura Beta (concentração em SOH alto e cauda de baterias degradadas).
2. Cada bateria recebe 8 cenários de SOC e temperatura por Latin Hypercube, totalizando 2.560 linhas por dataset.
3. A capacidade de referência é medida por uma descarga C20 simulada a 25 °C. O target é essa capacidade medida dividida por 60 Ah.
4. Capacidade e resistência são parcialmente desacopladas: `resistance_health = capacity_soh + ruído`, e os parâmetros R/C têm variação individual própria.
5. As features de pulso incluem resistência dinâmica/inicial, queda de tensão, recuperação e constante de tempo de relaxamento. As de cranking incluem mínimo de tensão, resistência efetiva, recuperação e constante de tempo.
6. As colunas de capacidade (`capacity_ah_*`, `soh_capacity_parameter`) e os identificadores não entram nos CSVs de features. Elas ficam apenas em `battery_metadata.csv`.
7. A divisão treino/validação/teste é feita por `battery_id` e estratificada por faixa de SOH: 220 baterias para treino, 50 para validação e 50 para teste.
8. A seleção do modelo usa apenas validação cruzada `StratifiedGroupKFold` (5 folds × 2 repetições) no conjunto de treino. O conjunto de teste é avaliado uma única vez, após a escolha.
9. O modelo escolhido é reajustado em treino + validação e avaliado no teste. Os intervalos de confiança são obtidos por bootstrap agrupado por bateria (2.000 reamostragens).

## Modelos comparados

- `DummyMean`: baseline trivial.
- `Ridge`: regressão linear regularizada com todas as features.
- `PhysicalRidge`: baseline físico que corrige a resistência medida por temperatura e SOC e ajusta uma regressão Ridge sobre termos polinomiais. Usa coeficientes exploratórios do simulador; serve como referência superior da relação R–SOH, não como modelo pronto para campo.
- `RandomForest`, `SVR_RBF`, `HistGradientBoosting`, `XGBoost`, `LightGBM` e `MLP` com early stopping agrupado por bateria.
- `CatBoost` e `K-Nearest Neighbors (KNN)`com três valores de k.


## Principais artefatos em `outputs/`

- Datasets: `pulse_dataset.csv`, `cranking_dataset.csv`, `combined_dataset.csv`, `battery_metadata.csv`.
- Resumos: `dataset_summary.csv`, `feature_summary.csv`, `soh_distribution.csv`, `split_manifest.csv`.
- Validação: `cv_folds_*.csv`, `cv_results_*.csv`, `val_results_*.csv`, `selected_models.csv`, `comparison_all.csv`.
- Teste: `test_metrics_*.csv`, `test_predictions_*.csv`, `battery_predictions_*.csv`.
- Análise de erro: `error_by_soh_*.csv`, `error_by_soc_*.csv`, `error_by_temperature_*.csv`, `feature_importance_*.csv`.
- Modelos: `*_selected_*.joblib`.
- Figuras: `predictions_*.png`, `residuals_*.png`.
