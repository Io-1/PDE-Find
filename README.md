# PDE-SINDy Pipeline

Minimal tooling for:

1. **Dataset generation** – simulate every PDE model listed in `config.yaml`.
2. **Optimizer sweeps** – run sparse-regression (STLSQ, SR3, …) sweeps, also driven by the same YAML.

---

## Quick Start

```bash
# 0 - install requirements
pip install -r requirements.txt

# 1 – check / edit the YAML
nano config.yaml          

# 2 – create synthetic data
python generate_datasets.py --config config.yaml   

# 3 – run hyper-parameter sweeps on those data sets
python run_sweep.py         --config config.yaml



All outputs are placed:
data/<model_name>/         # .npy files
results/<model_name>/      # sweeps.jsonl + metadata

---

Add or comment out full models with their settings to control 

seed: 42          # reproducible RNG
paths:
  data_dir:    data
  results_dir: results

models:
  heat_equation:      # key = class name in snake_case
    N_space: 64
    N_time: 32
    dt: 0.05
    alpha: 0.01

    library:          # SINDy library settings
      derivative_order: 4
      include_bias: true
      feature_presets: [all]

    optimizers:
      - name: STLSQ
        sweep_n_iter: 500
        sweep:
          threshold: [0.02, 8.0, 40]   # [min, max, points]
          alpha:     [1e-6, 1e-2, 40]
