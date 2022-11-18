import logging

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from active_learning_lab.utils.calibration import expected_calibration_error


METRIC_COLUMNS = [
    'train_acc', 'train_micro_precision', 'train_micro_recall', 'train_micro_f1',
    'train_macro_precision', 'train_macro_recall', 'train_macro_f1', 'train_ece_10',
    'test_acc', 'test_micro_precision', 'test_micro_recall', 'test_micro_f1',
    'test_macro_precision', 'test_macro_recall', 'test_macro_f1', 'test_ece_10'
]

COLUMNS = ['run_id', 'query_id', 'num_samples', 'query_time_sec', 'update_time_sec'] + \
          METRIC_COLUMNS


class MetricsTracker(object):

    NO_VALUE = -1

    def __init__(self):
        self.measured_metrics = pd.DataFrame(columns=COLUMNS)

    def track(self, run_id, query_id, num_labeled, run_results):

        times = [run_results.query_time, run_results.update_time]
        metrics_train = self._compute_metrics(run_results.y_train_true,
                                              run_results.y_train_pred,
                                              run_results.y_train_pred_proba)
        metrics_train_labeled = self._compute_metrics(run_results.y_train_subset_true,
                                                      run_results.y_train_subset_pred,
                                                      run_results.y_train_subset_pred_proba)
        metrics_test = self._compute_metrics(run_results.y_test_true,
                                             run_results.y_test_pred,
                                             run_results.y_test_pred_proba)

        logging.info(f'\tTest Acc: {metrics_test[0] * 100:>4.1f}\t'
                     f'TrainL Acc: {metrics_train_labeled[0] * 100:>4.1f}\t'
                     f'Train Acc: {metrics_train[0] * 100:>4.1f}')
        logging.info(f'\tTest ECE: {metrics_test[-1] * 100:>4.1f}\t'
                     f'TrainL ECE: {metrics_train_labeled[-1] * 100:>4.1f}\t'
                     f'Train ECE: {metrics_train[-1] * 100:>4.1f}')
        logging.info('')

        measured_metrics = [int(run_id), int(query_id), num_labeled] \
            + times + metrics_train + metrics_test
        self.measured_metrics.loc[len(self.measured_metrics)] = measured_metrics

    def track_train_history(self, query_id, train_history_dir, train_history):

        data = []

        selected_model = train_history.selected_model
        for i, entry in enumerate(train_history.metric_history):
            train_loss, train_acc, valid_loss, valid_acc = entry
            data.append([train_loss, train_acc, valid_loss, valid_acc, selected_model == i])

        df = pd.DataFrame(data, columns=['train_loss', 'train_acc', 'valid_loss', 'valid_acc',
                                         'is_selected_model'])

        output_file = Path(train_history_dir).joinpath(str(query_id) + '.csv')
        df.to_csv(output_file, index=False, header=True)

    @staticmethod
    def _compute_metrics(y_true, y_pred, y_pred_probas):

        if y_pred.shape[0] == 0:
            return [MetricsTracker.NO_VALUE] * 8
        else:
            y_pred_probas = np.amax(y_pred_probas, axis=1)
            return [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred, average='micro'),
                recall_score(y_true, y_pred, average='micro'),
                f1_score(y_true, y_pred, average='micro'),
                precision_score(y_true, y_pred, average='macro'),
                recall_score(y_true, y_pred, average='macro'),
                f1_score(y_true, y_pred, average='macro'),
                expected_calibration_error(y_pred, y_pred_probas, y_true)
            ]

    def write(self, output_file):
        self.measured_metrics = self.measured_metrics \
            .astype({'run_id': int, 'query_id': int, 'num_samples': int})
        self.measured_metrics.to_csv(output_file, index=False, header=True)

        return output_file

    def write_aggregate(self, output_file):
        gb = self.measured_metrics.groupby(['query_id'])
        df = gb[METRIC_COLUMNS].agg(['mean', 'std'], ddof=0)
        df.columns = df.columns.to_flat_index()
        df.columns = [tup[0] + '_' + tup[1] for tup in df.columns]
        df.to_csv(output_file, index=False, header=True)

        return output_file


class QueryTracker(object):

    def __init__(self, run_id):
        self.run_id = run_id

        self.query_data = {
            'initial_indices': None,
            'initial_labels': [],
            'queried_indices': [],
            'queried_labels': [],
            'queried_scores': []
        }

    def track_initial_indices(self, indices, labels):
        if self.query_data['initial_indices'] is None:
            self.query_data['initial_indices'] = indices.tolist()
            self.query_data['initial_labels'] = labels.tolist()
        else:
            raise ValueError('Initial indices can only bet set once')

    def track_queried_indices(self, indices, labels, scores):

        self.query_data['queried_indices'].append(indices.tolist())
        self.query_data['queried_labels'].append(labels.tolist())

        if scores is None:
            self.query_data['queried_scores'].append(scores)
        else:
            self.query_data['queried_scores'].append(scores.tolist())

    def write(self, output_path):

        if len(self.query_data['queried_scores']) > 0 and \
                all([score is None for score in self.query_data['queried_scores']]):
            self.query_data['queried_scores'] = None

        np.savez(output_path,
                 initial_indices=self.query_data['initial_indices'],
                 initial_labels=self.query_data['initial_labels'],
                 queried_indices=self.query_data['queried_indices'],
                 queried_labels=self.query_data['queried_labels'],
                 queried_scores=self.query_data['queried_scores'])

        return output_path


class RunResults(object):
    """
    Holds the results results of a single run.
    """
    def __init__(self, query_time, update_time, y_train_true, y_train_pred, y_train_pred_proba,
                 y_train_subset_true, y_train_subset_pred, y_train_subset_pred_proba,
                 y_test_true, y_test_pred, y_test_pred_proba):
        self.query_time = query_time
        self.update_time = update_time
        self.y_train_true = y_train_true
        self.y_train_pred = y_train_pred
        self.y_train_pred_proba = y_train_pred_proba
        self.y_train_subset_true = y_train_subset_true
        self.y_train_subset_pred = y_train_subset_pred
        self.y_train_subset_pred_proba = y_train_subset_pred_proba
        self.y_test_true = y_test_true
        self.y_test_pred = y_test_pred
        self.y_test_pred_proba = y_test_pred_proba
