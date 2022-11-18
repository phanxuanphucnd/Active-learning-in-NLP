import logging

from active_learning_lab.utils.numpy import get_class_histogram


def log_class_distribution(y, num_classes):
    logging.info('Class Distribution:')
    logging.info(get_class_histogram(y, num_classes, normalize=False))


def log_run_info(run_id, run_max, len_train, len_test):
    logging.info('#--------------------------------')
    logging.info('## Split: %d of %d', run_id, run_max)
    logging.info('##   Train %d / Test %d', len_train, len_test)
    logging.info('#--------------------------------')
