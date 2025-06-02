import numpy as np

def compute_pde_residual(
    data,
    dt,
    pde_lib,
    coefficients,
    finite_diff_method="forward"
):

    n_x, n_t, n_eq = data.shape

    if finite_diff_method == "forward":
        dUdt = (data[:, 1:, :] - data[:, :-1, :]) / dt  # shape (n_x, n_t-1, n_eq)
    else:
        raise NotImplementedError(
            f"finite_diff_method='{finite_diff_method}' not implemented."
        )

    X_array = pde_lib.transform(data)  # PDELibrary usually takes a list of arrays

    X_array = X_array[:, :-1, :]  # shape now (n_x, n_t-1, n_features)

    n_features = X_array.shape[-1]
    X_array_2d = X_array.reshape(-1, n_features)  # shape (n_x*(n_t-1), n_features)

    dUdt_2d = dUdt.reshape(-1, n_eq)  # shape (n_x*(n_t-1), n_eq)

    rmses = []
    for eq_idx in range(n_eq):
        coeff_eq = coefficients[eq_idx, :]  # discovered PDE for eq idx
        pred_t = X_array_2d @ coeff_eq  # shape (n_x*(n_t-1),)

        true_t = dUdt_2d[:, eq_idx]  # shape (n_x*(n_t-1),)

        resid = true_t - pred_t
        rmse = np.sqrt(np.mean(resid**2)) / np.sqrt(np.mean(true_t**2))
        rmses.append(rmse)
    mean_rmse = np.mean(rmses)
    return mean_rmse

def coefficient_error(identified_coefs, actual_coefs, mode='all_l2'):
    """
    identified_coefs : shape (n_features, m_equations)
    actual_coefs     : shape (n_features, m_equations) or (m_equations, n_features)
    mode             : str, 'all_l2' or 'per_equation'

    Returns:
      If 'all_l2': single float, the norm difference of all coefficients
      If 'per_equation': list of float, each PDE eq's difference
    """
    # Make shapes consistent
    if np.shape(actual_coefs) != np.shape(identified_coefs):
        # try transposing if the mismatch is just reversed
        if (np.shape(actual_coefs)[0] == np.shape(identified_coefs)[1]
            and np.shape(actual_coefs)[1] == np.shape(identified_coefs)[0]):
            actual_coefs = actual_coefs.T
        else:
            raise ValueError("Shapes differ and cannot be fixed by transpose!")
    
    if mode == 'all_l2':
        return np.linalg.norm(identified_coefs.ravel() - actual_coefs.ravel()) / np.linalg.norm(actual_coefs.ravel())
    elif mode == 'per_equation':
        n_eqs = identified_coefs.shape[1]
        errs = []
        for eq_idx in range(n_eqs):
            err_i = np.linalg.norm(identified_coefs[:, eq_idx] - actual_coefs[:, eq_idx])
            errs.append(err_i)
        return errs
    else:
        raise ValueError("Unknown mode. Must be 'all_l2' or 'per_equation'.")

def compute_classification_metrics(pred_coefs: np.ndarray, true_coefs: np.ndarray) -> dict:
    """
    Compute TP, TN, FP, FN, accuracy, precision, recall, and F1
    based on nonzero support in predicted vs true coefficients.
    """
    
    pred_mask = pred_coefs != 0
    true_mask = np.array(true_coefs) != 0
    
    TP = np.sum(pred_mask & true_mask)
    TN = np.sum(~pred_mask & ~true_mask)
    FP = np.sum(pred_mask & ~true_mask)
    FN = np.sum(~pred_mask & true_mask)
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": F1
    }
