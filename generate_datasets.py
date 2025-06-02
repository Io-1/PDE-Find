#!/usr/bin/env python3
"""
Generate and save 1D PDE datasets based on centralized config.yaml for model parameters & library settings.
Usage:
    python generate_datasets.py --config config.yaml --model gray_scott
    python generate_datasets.py --config config.yaml --model all
"""
import argparse
from pathlib import Path
import yaml
import json

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

from pde import ALL_MODELS  # mapping of model names to PDE classes


def load_config(path: Path) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate 1D PDE datasets from config.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to centralized YAML config file",
    )
    parser.add_argument(
        "--model",
        choices=list(ALL_MODELS.keys()) + ["all"],
        default="all",
        help="Which model(s) to generate ('all' or specific name)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_base = Path(cfg["paths"]["data_dir"])
    data_base.mkdir(parents=True, exist_ok=True)

    # Iterate over selected models
    model_list = list(cfg["models"].keys()) if args.model == "all" else [args.model]
    for name in model_list:
        print(f"{name}")
        sys_cfg = cfg["models"][name]
        cls = ALL_MODELS[name]

        # Build constructor parameters from defaults and config
        default_inst = cls()
        simulation_params = default_inst.simulation_params
        init_params = {k: sys_cfg.get(k, simulation_params[k]) for k in simulation_params}

        # Instantiate model with YAML-driven settings
        sys_model = cls(**init_params)         

        # Prepare output directory
        outdir = data_base / name
        outdir.mkdir(parents=True, exist_ok=True)

        # ---- Simulation ----
        data, dt, x = sys_model.simulate()
        np.save(outdir / "u_v.npy", data)
        np.save(outdir / "dt.npy", np.array(dt))
        np.save(outdir / "x.npy", x)

        # Save initial and boundary conditions
        ics = sys_model.initial_conditions()
        np.savez(outdir / "initial_conditions.npz", **ics)
        bcs = sys_model.boundary_conditions()
        with open(outdir / "boundary_conditions.json", 'w') as f:
            json.dump(bcs, f, indent=2)

        print(f"✅ '{name}' in {outdir.resolve()}")


if __name__ == "__main__":
    main()