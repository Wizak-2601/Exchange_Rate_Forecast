# EX-Dash: Exchange Rate Forecasting Dashboard

## 1) What This Project Is

This project is a complete time-series forecasting workflow for exchange-rate prediction, built around transformer-family sequence models and wrapped in an interactive Streamlit dashboard.

It combines:

- data preparation for sequential forecasting,
- model training and comparison across multiple architectures and hyperparameter settings,
- baseline benchmarking,
- final model evaluation on held-out data,
- and a UI (`ui/`) to explore experiments, leaderboard metrics, and forecast behavior.

### Core objective

Predict future exchange-rate values over multiple forecast horizons and compare deep learning models (Informer, Autoformer, Transformer baseline) against simpler baselines (for example ARIMA/naive references) using sMAPE-oriented evaluation.

### Main project components

- Training/evaluation pipeline: [`src/`](/Users/akarn/Documents/Exchange_curr_pred/src)
- Notebooks for EDA, experiments, and evaluation: [`notebooks/`](/Users/akarn/Documents/Exchange_curr_pred/notebooks)
- Final and intermediate artifacts (models + CSV outputs): [`notebooks/results/`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/results)
- UI dashboard: [`ui/`](/Users/akarn/Documents/Exchange_curr_pred/ui)


## 2) Setup Locally

### Prerequisites

- Python 3.10+ recommended
- `pip`
- (Optional but recommended) virtual environment

### Installation

1. Clone and enter the project:

```bash
git clone <your-repo-url>
cd Exchange_curr_pred
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the dashboard

From project root:

```bash
streamlit run ui/app.py
```

The dashboard includes:

- Home (project intro),
- Experiments Explorer,
- Model Leaderboard,
- Forecast Horizon.

### Data and result paths expected by UI

The UI loaders read from:

- [`notebooks/results/experiment_results.csv`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/results/experiment_results.csv)
- [`notebooks/results/autoformer_forecast.csv`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/results/autoformer_forecast.csv)
- [`notebooks/results/informer_forecast.csv`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/results/informer_forecast.csv)
- [`notebooks/results/transformer_forecast.csv`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/results/transformer_forecast.csv)

If you retrain models, keep these files updated (or adapt loaders in [`ui/utils/load_data.py`](/Users/akarn/Documents/Exchange_curr_pred/ui/utils/load_data.py)).


## 3) Project Structure

```text
Exchange_curr_pred/
├── data/
│   └── raw/                          # train/validation/test CSVs
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── Informer_and_autoformer.ipynb
│   ├── baseline_models.ipynb
│   ├── result_evaluation.ipynb
│   └── results/                      # experiment outputs + forecast CSVs + final tables
├── results/
│   ├── saved_models/                 # checkpointed experiments
│   └── final_models/                 # final selected model weights
├── src/
│   ├── configs/                      # YAML config templates
│   ├── data/                         # loading, windowing, dataloaders
│   ├── models/                       # informer / autoformer / transformer builders
│   ├── training/                     # train loop + early stopping
│   ├── evaluation/                   # metrics + baseline evaluators
│   ├── experiments/                  # experiment runner
│   └── utils/                        # config + plotting + logging helpers
├── ui/
│   ├── app.py                        # Streamlit entrypoint
│   ├── pages/                        # dashboard pages
│   └── utils/                        # UI data loaders
├── requirements.txt
└── README.md
```


## 4) How to Retrain Models

This project currently uses a notebook-driven experiment flow, with reusable Python modules underneath:

- config loading: [`src/utils/config.py`](/Users/akarn/Documents/Exchange_curr_pred/src/utils/config.py)
- experiment execution: [`src/experiments/run_experiment.py`](/Users/akarn/Documents/Exchange_curr_pred/src/experiments/run_experiment.py)
- result logging: [`src/utils/results_logger.py`](/Users/akarn/Documents/Exchange_curr_pred/src/utils/results_logger.py)

### Recommended workflow (notebook)

1. Open:
   - [`notebooks/Informer_and_autoformer.ipynb`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/Informer_and_autoformer.ipynb)
   - [`notebooks/baseline_models.ipynb`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/baseline_models.ipynb)
2. Set/adjust model and training parameters (model type, sequence length, horizon, heads, dropout, etc.).
3. Run cells that call `load_config(...)`, `run_experiment(config)`, and `save_results(res)`.
4. Confirm outputs are updated in:
   - `results/saved_models/` (checkpoints),
   - `notebooks/results/experiment_results.csv` (aggregated experiment metrics),
   - `notebooks/results/*_forecast.csv` (forecast outputs used in dashboard).

### Programmatic retrain example (script-style)

```python
from src.utils.config import load_config
from src.experiments.run_experiment import run_experiment
from src.utils.results_logger import save_results

config = load_config(
    "src/configs/base.yaml",
    overrides={
        "model_type": "autoformer",  # or "informer" / "transformer"
        "seq_len": 96,
        "pred_len": 199,
        "n_heads": 5,
        "dropout": 0.2,
        "epochs": 20,
        "batch_size": 16,
        "lr": 1e-4,
        "univariate": False,
        "residual": True,
    },
)

result = run_experiment(config)
save_results(result, filepath="notebooks/results/experiment_results.csv")
print(result)
```

### After retraining

1. Regenerate/update final comparison artifacts in your evaluation notebook:
   - [`notebooks/result_evaluation.ipynb`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/result_evaluation.ipynb)
2. Ensure dashboard source files still point to your latest artifacts:
   - [`ui/utils/load_data.py`](/Users/akarn/Documents/Exchange_curr_pred/ui/utils/load_data.py)
3. Restart the UI:

```bash
streamlit run ui/app.py
```


## 5) Final Results

The following values come from the current result artifacts in this repo:

- [`notebooks/results/experiment_results.csv`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/results/experiment_results.csv)
- [`notebooks/results/final_model_comparison_table.csv`](/Users/akarn/Documents/Exchange_curr_pred/notebooks/results/final_model_comparison_table.csv)

### Experiment sweep summary

- Total deep-learning experiment runs: **68**
- Best run across `experiment_results.csv`:
  - **Model**: Informer
  - **Config**: `seq_len=96`, `pred_len=24`, `n_heads=3`, `dropout=0.1`, `d_model=60`
  - **model_smape**: **0.08538**
  - **naive_smape**: **1.85761**

Mean `model_smape` across runs:

- Autoformer: **0.17319**
- Informer: **0.19216**

### Final comparison table (validation/test sMAPE)

| Model | Validation sMAPE | Test sMAPE | Run Name |
|---|---:|---:|---|
| ARIMA | 0.62449 | 0.70541 | ARIMA |
| Transformer | 0.18726 | 0.46254 | transformer_s192_p199_h4_k23_d0.1 |
| Informer | 0.20561 | 0.43261 | informer_s96_p199_h3_k23_d0.1 |
| Autoformer | 0.18020 | 0.43486 | autoformer_s96_p199_h5_k23_d0.2 |

### Interpretation

- Deep sequence models significantly outperform ARIMA on test sMAPE.
- Informer and Autoformer are close on test performance, with Informer slightly lower test sMAPE in the final comparison table.
- Autoformer shows the best average `model_smape` over the broader experiment sweep.


## 6) Suggested Next Steps

- Add confidence interval calibration metrics to leaderboard (coverage and sharpness).
- Track all experiments with a consistent run registry (single canonical CSV schema).
- Add automated tests for loaders and result integrity checks before launching UI.
