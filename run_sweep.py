#!/usr/bin/env python3
"""
Simplified sweep runner using a single config.yaml for model, library, and optimizer settings.
"""
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import jsonlines
import zipfile
import importlib
from datetime import datetime

from pysindy import PDELibrary
from pysindy import FiniteDifference, SmoothedFiniteDifference, SpectralDerivative

from library_functions import get_functions_and_naming_functions
import library_functions as lf
from sweep import sweep_optimizers
from pde import ALL_MODELS 
from scipy.ndimage import gaussian_filter1d


def load_config(path: Path) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweeps for SINDy PDE models using unified config."
    )
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
        help="Model to sweep ('all' or specific name)",
    )
    args = parser.parse_args()

    # Load settings
    cfg = load_config(args.config)
    global_seed = cfg.get("seed", 0)
    np.random.seed(global_seed)  # reproducible random sweep

    data_base = Path(cfg["paths"]["data_dir"])
    result_base = Path(cfg["paths"]["results_dir"])

    # Global defaults for search
    global_n_iter = cfg.get("sweep_n_iter", 250)
    global_search_type = cfg.get("search_type", "random")

    # Determine which models to run
    model_list = list(ALL_MODELS.keys()) if args.model == "all" else [args.model]

    for name in model_list:
        print(f"--- Sweeping model: {name} ---")
        sys_cfg = cfg["models"][name]

        # Instantiate model with config-driven parameters
        default_inst = ALL_MODELS[name]()
        simulation_params = default_inst.simulation_params
        init_params = {k: sys_cfg.get(k, simulation_params[k]) for k in simulation_params}

        model = ALL_MODELS[name](**init_params)
        model_data_dir = data_base / name

        # Load simulation data
        data = np.load(model_data_dir / "u_v.npy")    # (n_x, n_t, n_eq)
        dt = float(np.load(model_data_dir / "dt.npy"))
        x = np.load(model_data_dir / "x.npy")
        n_eq = data.shape[2]

        noise_pct = 0.00

        if noise_pct > 0:
            scale = noise_pct * (data.max(axis=(0,1)) - data.min(axis=(0,1)))
            data = data + np.random.randn(*data.shape) * scale[np.newaxis, np.newaxis, :]

            data = gaussian_filter1d(data, sigma=1, axis=0)

        eps = 1e-8
        data = np.clip(data, eps, None)

        lf.generate_registry(n_eq, max_degree=4)
        lib_config = {k: sys_cfg["library"][k] for k in sys_cfg["library"] if k not in ["feature_functions", "feature_presets", "drop_derivatives"]}
        presets = sys_cfg["library"].get("feature_presets", [])
        fns = sys_cfg["library"].get("feature_functions", [])
        funcs, name_funcs = lf.get_functions_and_naming_functions(presets, fns)

        deriv_order = sys_cfg["library"].get("derivative_order", 2)
        dm_cfg = sys_cfg["library"].get("differentiation_method")
        if dm_cfg is None:
            lib_config["differentiation_method"] = FiniteDifference
        else:
            name_diff = dm_cfg["name_diff"]
            if name_diff == "SmoothedFiniteDifference":
                lib_config["differentiation_method"] = SmoothedFiniteDifference
            elif name_diff == "SpectralDerivative":
                lib_config["differentiation_method"] = SpectralDerivative
            else:
                raise ValueError(f"Unknown method {name_diff}")
        
        lib_config["diff_kwargs"] = sys_cfg["library"].get("diff_kwargs", {})

        # Build and fit library using model's method
        pde_lib = PDELibrary(**lib_config,
            library_functions = funcs,
            function_names = name_funcs,
            spatial_grid = x,
        )
        pde_lib.fit(data)
        X3d = pde_lib.transform(data)                   # (n_x, n_t, F)
        feature_names = pde_lib.get_feature_names()

        drop_derivatives = sys_cfg["library"].get("drop_derivatives", [])
        

        rename_map = {
            "x0_1":  "uₓ",
            "x0_11": "uₓₓ",
            "x0_111": "uₓₓₓ",
            "x0_1111": "uₓₓₓₓ",
            "x0_11111": "uₓₓₓₓₓ",
            "x0_111111": "uₓₓₓₓₓₓ",

            "x1_1": "vₓ",
            "x1_11": "vₓₓ",
            "x1_111": "vₓₓₓ",
            "x1_1111": "vₓₓₓₓ",
            "x1_11111": "vₓₓₓₓₓ",
            "x1_111111": "vₓₓₓₓₓₓ",

        }

        feature_names = pde_lib.get_feature_names()

        # Build a new list, replacing any key found in rename_map:
        feature_names = [
            rename_map.get(name, name) 
            for name in feature_names
        ]
        drop_derivatives = [
            rename_map.get(name, name)
            for name in drop_derivatives
        ]
        to_drop = [i for i, d in enumerate(feature_names) if d in drop_derivatives]
        pde_lib._feature_names = feature_names

        if to_drop:
            X3d = np.delete(X3d, to_drop, axis=2)
            feature_names = [n for i,n in enumerate(feature_names) if i not in to_drop]
        
        # Prepare regression matrices
        dUdt3d = (data[:, 1:, :] - data[:, :-1, :]) / dt  # (n_x, n_t-1, n_eq)

        n_x, n_t, F = X3d.shape
        _, _, E = dUdt3d.shape
        X = X3d[:, :-1, :].reshape(-1, F)  # (N, F)
        y = dUdt3d.reshape(-1, E)          # (N, E)

        # Load ground-truth coefficients

        target_coefs = model.get_target_coefs(feature_names)
        if to_drop:
            target_coefs = [[n for i, n in enumerate(target_coefs[eq])] for eq in range(len(target_coefs))]
        else:
            target_coefs = [[n for i, n in enumerate(target_coefs[eq])] for eq in range(len(target_coefs))]

        # Build optimizer configurations from YAML
        optimizer_configs = []
        for opt in sys_cfg["optimizers"]:
            # Determine search type and iteration budget
            search_type = opt.get("search_type", global_search_type)
            n_iter = opt.get("sweep_n_iter", global_n_iter)

            # Create linear space or explicit array for each sweep param
            param_ranges = {}
            for p, triple in opt.get("sweep", {}).items():
                if len(triple) == 3:
                    start, stop, steps = triple
                    grid = np.linspace(float(start), float(stop), int(steps))
                else:
                    grid = np.array(triple, dtype=float)
                param_ranges[p] = grid

            # fixed params: everything except meta keys
            fixed = {
                k: v
                for k, v in opt.items()
                if k not in ("name", "sweep", "sweep_n_iter", "search_type")
            }

            optimizer_configs.append({
                "optimizer":    opt["name"],
                "search_type":  search_type,
                "n_iter":       n_iter,
                "random_state": global_seed,
                "param_ranges": param_ranges,
                "fixed_params": fixed,
            })

        # Run hyperparameter sweep
        print("Launching sweep...")
        results = sweep_optimizers(
            X, y, X3d, data, dt,
            optimizer_configs, target_coefs,
            n_jobs=cfg.get("n_jobs", 4)
        )
        print(f"Completed {len(results)} trials.")

        # Save results
        out_dir = result_base / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(out_dir / "sweeps.jsonl", mode='w') as writer:
            for r in results:
                writer.write(r)
        with open(out_dir / "target_coefs.json", 'w') as f:
            json.dump(target_coefs, f, indent=2)
        with open(out_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)

        archive_dir = result_base / "archives"
        archive_dir.mkdir(exist_ok=True)

        existing = list(archive_dir.glob(f"{name}_run_*.zip"))
        run_id  = len(existing) + 1

        zip_path = archive_dir / f"{name}_run_{run_id:03d}.zip"
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.write(args.config, arcname="config.yaml")
            for fn in ("sweeps.jsonl", "target_coefs.json", "feature_names.json"):
                z.write(out_dir / fn, arcname=fn)

        print(f"Results saved under: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
