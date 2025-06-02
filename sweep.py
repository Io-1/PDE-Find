import numpy as np
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import r2_score
from metrics import compute_classification_metrics


def run_batch(batch_args):
    """
    Run a batch of trials sequentially using the matrix-based runner.
    """
    return [run_trial_matrix(*args) for args in batch_args]


def sweep_optimizers(
    X, y, X3d, data, dt,
    optimizer_configs, target_coefs, n_jobs=-1
):
    """
    Perform hyperparameter sweeps over matrix-based trials, in parallel batches.
    Returns list of all trial results.
    """
    # Determine number of workers
    if n_jobs is None or n_jobs == 0:
        workers = 1
    elif n_jobs < 0:
        workers = cpu_count()
    else:
        workers = n_jobs
    workers = max(1, workers)

    # Build list of trial arguments
    args_list = []
    for config in optimizer_configs:
        stype = config.get("search_type", "grid")
        if stype == "grid":
            iterator = ParameterGrid(config["param_ranges"])
        elif stype == "random":
            iterator = ParameterSampler(
                config["param_ranges"],
                n_iter=config.get("n_iter", 100),
                random_state=config.get("random_state", None)
            )
        else:
            raise ValueError(f"Unknown search_type={stype!r}")

        for params in iterator:
            full_params = {**params, **config.get("fixed_params", {})}
            args_list.append((
                config["optimizer"],
                full_params,
                X, y, X3d, data, dt, target_coefs
            ))

    total = len(args_list)
    if total == 0:
        return []

    # Split into batches for lower overhead
    batch_size = max(1, total // workers)
    batches = [args_list[i:i+batch_size] for i in range(0, total, batch_size)]

    # Execute batches
    results = []
    if workers == 1:
        for batch in tqdm(batches, desc="Batches", total=len(batches)):
            results.extend(run_batch(batch))
    else:
        with Pool(workers) as pool:
            for batch_res in tqdm(
                pool.imap(run_batch, batches),
                desc="Batches",
                total=len(batches)
            ):
                results.extend(batch_res)

    return results


def run_trial_matrix(
    optimizer_name, params,
    X, y, X3d, data, dt, target_coefs
):
    """
    Run one SINDy optimizer on pre-flattened matrices, computing metrics without further transforms.
    """
    # Instantiate optimizer
    Optim = getattr(__import__('pysindy').optimizers, optimizer_name)
    ensemble = params.pop("ensemble", None)
    ensemble_params = params.pop("ensemble_params", None)
    optimizer = Optim(**params)
    if ensemble:
        Ensem = getattr(__import__('pysindy').optimizers, "EnsembleOptimizer")
        optimizer = Ensem(**ensemble_params, opt = optimizer)

    # Dimensions
    F, E = X.shape[1], y.shape[1]

    # Fit each equation separately
    precision = 1e-5
    coefs = np.zeros((F, E))
    for eq in range(E):
        optimizer.fit(X, y[:, eq])
        coefs[:, eq] = optimizer.coef_

    coefs[np.abs(coefs) < precision] = 0
    # Predictions and ground truth
    pred = X.dot(coefs)
    true = y

    # Sparsity: count nonzeros per equation
    sparsity_total = int((coefs != 0).sum())
    sparsity_per_eq = np.count_nonzero(coefs, axis=0)

    # Residual: normalized RMSE per equation
    rmse_eq = np.sqrt(np.mean((true - pred)**2, axis=0))
    norm_eq = np.sqrt(np.mean(true**2, axis=0))
    residuals_eq = rmse_eq / norm_eq
    residual = float(np.mean(residuals_eq))

    # Score: multi-output R^2
    score = float(r2_score(true, pred, multioutput='uniform_average'))

    # Classification metrics on support
    class_metrics = compute_classification_metrics(coefs, target_coefs)
    class_metrics = {
        k: int(v) if isinstance(v, (bool, np.integer)) else float(v)
        for k, v in class_metrics.items()
    }
    norm_target = np.linalg.norm(target_coefs, axis=0)
    norm_true = np.linalg.norm(coefs, axis=0)
    norm_diff = np.linalg.norm(coefs - target_coefs, axis=0)
    coef_errs_eq = norm_diff / (norm_true + norm_target + 1e-8)
    coef_err = float(np.mean(coef_errs_eq))


    # Build result dict dynamically
    result = {
        'optimizer': optimizer_name,
        **params,
        'sparsity': sparsity_total,
        'residual': residual,
        'score': score,
        'coef_err': coef_err,
        **class_metrics,
    }

    # Add per-equation metrics and coefficients
    for i in range(E):
        result[f'residual{i}'] = float(residuals_eq[i])
        result[f'sparsity{i}'] = int(sparsity_per_eq[i])
        result[f'coef_err{i}'] = float(coef_errs_eq[i])
        result[f'coefs{i}'] = coefs[:, i].tolist()

    return result

