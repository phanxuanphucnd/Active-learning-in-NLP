import numpy as np


def get_class_histogram(y, num_classes, normalize=True):
    ind, counts = np.unique(y, return_counts=True)
    ind_set = set(ind)

    histogram = np.zeros(num_classes)
    for i, c in zip(ind, counts):
        if i in ind_set:
            histogram[i] = c

    if normalize:
        return histogram / histogram.sum()

    return histogram.astype(int)
