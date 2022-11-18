import numpy as np


def expected_calibration_error(y_pred, proba, y_true, n_bins=10):

    intervals = np.linspace(0, 1, n_bins+1)
    accuracy = (y_pred == y_true).astype(int)

    num_predictions = y_pred.shape[0]
    error = 0

    for lower, upper in zip(intervals[:-1], intervals[1:]):
        mask = np.logical_and(proba > lower, proba <= upper)
        bin_size = mask.sum()
        if bin_size > 0:
            proba_bin_mean = proba[mask].mean()
            acc_bin_mean = accuracy[mask].mean()

            error += bin_size / num_predictions * np.abs(acc_bin_mean - proba_bin_mean)

    return error
