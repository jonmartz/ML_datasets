import numpy as np


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    n_total = len(y_true)
    actual_confirmed = y_true == 'Confirmed'
    n_actual_confirmed = sum(actual_confirmed)
    n_actual_rejected = sum(y_true == 'Rejected')

    pred_confirmed = y_pred == 'Confirmed'
    n_true_confirmed = sum(pred_confirmed & actual_confirmed)
    n_true_rejected = sum(~pred_confirmed & ~actual_confirmed)
    n_pred_confirmed = n_actual_rejected - n_true_rejected + n_true_confirmed
    n_pred_rejected = n_actual_confirmed - n_true_confirmed + n_true_rejected

    result = {
        'accuracy': (n_true_confirmed + n_true_rejected) / n_total,
        'Confirmed precision': n_true_confirmed / n_pred_confirmed if n_pred_confirmed else 1.0,
        'Confirmed recall': n_true_confirmed / n_actual_confirmed if n_actual_confirmed else 1.0,
        'Rejected precision': n_true_rejected / n_pred_rejected if n_pred_rejected else 1.0,
        'Rejected recall': n_true_rejected / n_actual_rejected if n_actual_rejected else 1.0,
    }

    matrix = f"""
    Confirmed\tRejected
    {n_true_confirmed}\t{n_pred_rejected - n_true_rejected}
    {n_pred_confirmed - n_true_confirmed}\t{n_true_rejected}
    """
    print(matrix)

    for k, v in result.items():
        print(f'\t{k}: {v}')