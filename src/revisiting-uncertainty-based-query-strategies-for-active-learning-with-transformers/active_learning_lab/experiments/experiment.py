import glob
import logging

import mlflow
import torch

import numpy as np
import pandas as pd

from functools import partial
from pathlib import Path

from sklearn.metrics import auc

from small_text.active_learner import PoolBasedActiveLearner
from small_text.integrations.pytorch.classifiers import PytorchClassifier

from active_learning_lab.experiments.logging import (
    log_class_distribution,
    log_run_info
)
from active_learning_lab.experiments.tracking import (
    MetricsTracker,
    QueryTracker,
    RunResults
)
from active_learning_lab.experiments.query_strategies import query_strategy_from_str

from active_learning_lab.utils.data import get_validation_set
from active_learning_lab.utils.experiment import (
    free_resources_fix,
    set_random_seed
)
from active_learning_lab.utils.time import measure_time


INITIALIZATION_NUM_INSTANCES_DEFAULT = 25


class ExperimentConfig(object):
    def __init__(self, runs, num_queries, query_size):
        self.runs = runs
        self.num_queries = num_queries
        self.query_size = query_size


class ClassificationConfig(object):
    def __init__(self, classifier_name, classifier_factory, classifier_kwargs=None,
                 self_training=False, incremental_training=False,
                 validation_set_size=0.1):
        self.classifier_name = classifier_name

        if classifier_kwargs is None:
            self.classifier_kwargs = dict()
        else:
            self.classifier_kwargs = classifier_kwargs

        self.classifier_factory = classifier_factory
        self.self_training = self_training
        self.incremental_training = incremental_training
        self.validation_set_size = validation_set_size


class DatasetConfig(object):

    def __init__(self, dataset_name, dataset_kwargs, train_raw=None, test_raw=None):
        self.dataset_name = dataset_name
        self.dataset_kwargs = dataset_kwargs
        self.train_raw = train_raw
        self.test_raw = test_raw


class ActiveLearningExperimentResult(object):

    def __init__(self, artifacts):
        self.artifacts = artifacts


class ActiveLearningExperiment(object):

    def __init__(self, exp_cfg, classification_cfg, dataset_config, initialization_strategy,
                 initialization_strategy_kwargs, train, tmp_dir, query_strategy=None,
                 query_strategy_kwargs=dict(),
                 shared_initialization=True, gpu=None):

        self.classification_args = classification_cfg

        self.num_classes = np.unique(train.y).shape[0]

        self.initialization_strategy = initialization_strategy
        self.initialization_strategy_kwargs = initialization_strategy_kwargs
        self.shared_initialization = shared_initialization

        logging.info(f'Initialization: {str(self.initialization_strategy)}, '
                     f'shared={str(shared_initialization)}')

        self.train = train
        self.tmp_dir = Path(tmp_dir)

        self.query_strategy = query_strategy
        if isinstance(self.query_strategy, str):
            self.query_strategy = query_strategy_from_str(self.query_strategy,
                                                          query_strategy_kwargs)
        logging.info(f'Query strategy: {str(self.query_strategy)}')
        self.query_strategy_kwargs = query_strategy_kwargs
        self.dataset_config = dataset_config

        logging.info(f'train dataset: length={len(train)}, '
                     f'num_classes={self.num_classes}, type={str(type(train))}')

        self.x_ind_init = None

        self.exp_cfg = exp_cfg
        self.gpu = gpu

        self.incremental_training = classification_cfg.classifier_kwargs.get('incremental_training', False)
        if 'incremental_training' in classification_cfg.classifier_kwargs:
            del classification_cfg.classifier_kwargs['incremental_training']

        self.metrics_tracker = MetricsTracker()

    def run(self, train_set, test_set):
        artifacts = self._pre_experiment(train_set, test_set)

        # The seeds for the single runs are dependendent on the "main seed".
        # We fix them here so that the condition for every run is exactly the same.
        seeds = [np.random.randint(2*+32) for _ in range(self.exp_cfg.runs)]

        for run_id in range(1, self.exp_cfg.runs + 1):

            log_run_info(run_id, self.exp_cfg.runs, len(train_set), len(test_set))

            self._pre_run(run_id, train_set, test_set)
            artifacts += ActiveLearningRun(
                self.exp_cfg,
                self.tmp_dir,
                run_id,
                seeds[run_id-1],
                self.classification_args,
                self.dataset_config,
                self.num_classes,
                self.x_ind_init,
                self.metrics_tracker,
                self.query_strategy,
                self.query_strategy_kwargs,
                self.initialization_strategy,
                self.initialization_strategy_kwargs
            ).execute(run_id, train_set, test_set)
            self._post_run()

        return self._post_experiment(artifacts)

    def _pre_experiment(self, train_set, test_set):

        np.savez(self.tmp_dir.joinpath('train_labels.npz'), vectors=train_set.y)
        np.savez(self.tmp_dir.joinpath('test_labels.npz'), vectors=test_set.y)

        artifacts = [('train_labels.npz', self.tmp_dir.joinpath('train_labels.npz')),
                     ('test_labels.npz', self.tmp_dir.joinpath('test_labels.npz'))]

        return artifacts

    def _pre_run(self, run_id, train_set, test_set):
        pass

    def _post_run(self):
        if torch.cuda.is_available():
            free_resources_fix()

    def _post_experiment(self, artifacts):

        results_file = self.metrics_tracker.write(self.tmp_dir.joinpath('results.csv').resolve())
        results_agg_file = self.metrics_tracker.write_aggregate(
            self.tmp_dir.joinpath('results_agg.csv').resolve())

        artifacts += [('results.csv', results_file), ('results_agg.csv', results_agg_file)]
        artifacts = auc_metrics(self.tmp_dir, self.metrics_tracker, artifacts)

        return ActiveLearningExperimentResult(artifacts)


class ActiveLearningRun(object):

    def __init__(self, exp_args, tmp_dir, run_id, seed, classification_args, dataset_config,
                 num_classes,
                 x_ind_init, metrics_tracker, query_strategy, query_strategy_kwargs,
                 initialization_strategy, initialization_strategy_kwargs):

        self.exp_args = exp_args
        self.tmp_dir = tmp_dir
        self.seed = seed

        self.classification_args = classification_args
        self.dataset_config = dataset_config
        self.num_classes = num_classes
        self.x_ind_init = x_ind_init
        self.metrics_tracker = metrics_tracker

        self.query_strategy = query_strategy
        self.query_strategy_kwargs = query_strategy_kwargs

        self.initialization_strategy = initialization_strategy
        self.initialization_strategy_kwargs = initialization_strategy_kwargs

        self.query_dir = self.tmp_dir.joinpath('run_' + str(run_id))
        self.query_dir.mkdir(parents=True)

        self.train_history_dir = self.query_dir.joinpath('train_history')
        self.train_history_dir.mkdir()

    def execute(self, run_id, train_set, test_set):
        set_random_seed(self.seed)
        active_learner, y_init = self._get_initialized_active_learner(train_set)

        if self.classification_args.classifier_name == 'svm':
            x_indices_validation = None
        else:
            x_indices_validation = get_validation_set(
                y_init,
                validation_set_size=self.classification_args.validation_set_size)

        # Initial evaluation
        ind, run_results = self.run_initial_evaluation(active_learner, train_set,
                                                       x_indices_validation, test_set)
        self.metrics_tracker.track(run_id, 0, len(active_learner.x_indices_labeled), run_results)

        self.query_tracker = QueryTracker(run_id)
        self.query_tracker.track_initial_indices(self.x_ind_init, y_init)

        #
        # [!] This is the main loop
        #
        for q in range(1, self.exp_args.num_queries+1):
            ind, scores, run_results = self.run_query(active_learner, q, run_id,
                                                      train_set, test_set,
                                                      self.query_strategy_kwargs)
            y_train = train_set.y
            self.post_query(active_learner, ind, q, run_id, run_results, scores, y_train)

        return self._create_artifacts()

    def post_query(self, active_learner, ind, q, run_id, run_results, scores, y_train):

        queried_labels = np.array([y_train[i] for i in ind])

        # track query
        self.query_tracker.track_queried_indices(ind, queried_labels, scores)
        self.metrics_tracker.track(run_id, q, len(active_learner.x_indices_labeled),
                                   run_results)

        # track metrics
        if hasattr(active_learner.classifier, 'train_history'):
            if active_learner.classifier.train_history is not None:
                self.metrics_tracker.track_train_history(
                    q, self.train_history_dir, active_learner.classifier.train_history)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _get_initialized_active_learner(self, train_set):

        active_learner = PoolBasedActiveLearner(
            self.classification_args.classifier_factory,
            self.query_strategy,
            train_set,
            incremental_training=self.classification_args.incremental_training)

        strategy = self.initialization_strategy
        num_samples = self.initialization_strategy_kwargs.get('num_instances',
                                                              INITIALIZATION_NUM_INSTANCES_DEFAULT)
        if self.x_ind_init is None or self.shared_initialization is False:
            self.x_ind_init = get_initial_indices(train_set, strategy, num_samples)

        y_train = train_set.y
        y_init = np.array([y_train[i] for i in self.x_ind_init])
        active_learner.initialize_data(self.x_ind_init, y_init, retrain=False)

        if isinstance(active_learner.classifier, PytorchClassifier):
            active_learner.classifier.model = active_learner.classifier.model.cuda()

        return active_learner, y_init

    def _create_artifacts(self):

        queried_indices_file_path = self.query_dir.joinpath('queries.npz')
        self.query_tracker.write(queried_indices_file_path)

        artifacts = [('queries.npz', queried_indices_file_path)]

        for f in glob.glob(str(self.query_dir) + '/**/*', recursive=True):
            artifacts.append((f, self.query_dir.joinpath(f)))

        return artifacts

    def run_initial_evaluation(self, active_learner, train_set, x_indices_validation, test_set):

        y_init = train_set.y[self.x_ind_init]
        active_learner.initialize_data(self.x_ind_init, y_init, retrain=False)

        # Update (i.e. retrain in this case)
        update_time = measure_time(
            partial(active_learner._retrain, x_indices_validation=x_indices_validation),
            has_return_value=False)

        # Evaluation
        y_train_pred, y_train_proba = active_learner.classifier.predict(train_set,
                                                                        return_proba=True)
        if self.x_ind_init.shape[0] == 0:
            y_train_labeled_pred, y_train_labeled_proba = np.array([], int), np.array([], float)
        else:
            y_train_labeled_pred, y_train_labeled_proba = active_learner.classifier.predict(
                train_set[self.x_ind_init], return_proba=True
            )

        y_test_pred, y_test_proba = active_learner.classifier.predict(test_set, return_proba=True)

        np.savez(str(self.query_dir.joinpath('predictions_0.npz')),  # run_id starts at 1,
                 train_predictions=y_train_pred,                      # so 0 is the initial
                 train_proba=y_train_proba,                           # value
                 test_predictions=y_test_pred,
                 test_proba=y_test_proba)

        run_results = RunResults(0, update_time,
                                 train_set.y, y_train_pred, y_train_proba,
                                 train_set[self.x_ind_init].y, y_train_labeled_pred, y_train_labeled_proba,
                                 test_set.y, y_test_pred, y_test_proba)

        return active_learner.x_indices_labeled, run_results

    def run_query(self, active_learner, q, run_id, train_set, test_set,
                  _query_strategy_kwargs=dict()):

        logging.info('## Run: %d / Query: %d', run_id, q)
        x_indices_labeled = active_learner.x_indices_labeled

        if x_indices_labeled.shape[0] == 0:
            y_train_labeled_true = np.array([], dtype=int)
        else:
            y_train_labeled_true = train_set[x_indices_labeled].y

        query_func = partial(active_learner.query, num_samples=self.exp_args.query_size)
        query_time, ind = measure_time(query_func)

        log_class_distribution(active_learner.y, self.num_classes)

        query_strategy = active_learner.query_strategy
        scores = query_strategy.scores_ if hasattr(query_strategy, 'scores_') else None

        # Update
        if not active_learner.incremental_training and hasattr(active_learner.classifier, 'model'):
            active_learner.classifier.model.zero_grad(set_to_none=True)
            active_learner.classifier.model = None

        free_resources_fix()

        y_true_update = train_set[ind].y
        indices_labeled_and_update = np.concatenate((active_learner.x_indices_labeled, ind))

        if self.classification_args.classifier_name == 'svm':
            x_indices_validation = None
        else:
            x_indices_validation = get_validation_set(
                train_set[indices_labeled_and_update].y,
                validation_set_size=self.classification_args.validation_set_size
            )

        update_func = partial(active_learner.update, y_true_update,
                              x_indices_validation=x_indices_validation)
        update_time = measure_time(update_func, has_return_value=False)
        log_class_distribution(active_learner.y, self.num_classes)

        # Evaluation
        y_train_pred, y_train_proba = active_learner.classifier.predict(train_set,
                                                                        return_proba=True)
        y_train_labeled_pred, y_train_labeled_proba = active_learner.classifier.predict(
            train_set[x_indices_labeled], return_proba=True
        )

        y_test_pred, y_test_proba = active_learner.classifier.predict(test_set, return_proba=True)

        np.savez(str(self.query_dir.joinpath(f'predictions_{q}.npz')),
                 train_predictions=y_train_pred,
                 train_proba=y_train_proba,
                 test_predictions=y_test_pred,
                 test_proba=y_test_proba)

        run_results = RunResults(query_time, update_time,
                                 train_set.y, y_train_pred, y_train_proba,
                                 y_train_labeled_true, y_train_labeled_pred, y_train_labeled_proba,
                                 test_set.y, y_test_pred, y_test_proba)

        return ind, scores, run_results


def auc_metrics(tmp_dir, metrics_tracker, artifacts):

    df_auc = compute_auc_from_metrics_df(metrics_tracker.measured_metrics)
    auc_csv = tmp_dir.joinpath('auc.csv').resolve()
    df_auc.to_csv(auc_csv, index=False, header=True)
    artifacts += [('auc.csv', auc_csv)]

    df_auc_acc = df_auc.groupby(lambda _: True).agg(['mean', 'std'], ddof=0)
    auc_agg_csv = tmp_dir.joinpath('auc_agg.csv').resolve()
    df_auc_acc.columns = df_auc_acc.columns.to_flat_index()
    df_auc_acc.columns = [tup[0] + '_' + tup[1] for tup in df_auc_acc.columns]
    df_auc_acc.to_csv(auc_agg_csv, index=False, header=True)
    artifacts += [('auc_agg.csv', auc_csv)]

    aggregate_metrics_track_and_log(df_auc, metrics_tracker)

    return artifacts


def aggregate_metrics_track_and_log(df_auc, metrics_tracker):

    auc_test_acc_mean = df_auc['auc_test_acc'].mean()
    auc_test_acc_std = df_auc['auc_test_acc'].std(ddof=0)
    auc_test_micro_f1_mean = df_auc['auc_test_micro_f1'].mean()
    auc_test_micro_f1_std = df_auc['auc_test_micro_f1'].std(ddof=0)

    logging.info('#--------------------------------')
    logging.info(f'AUC: {auc_test_acc_mean:.4f} (+/- {auc_test_acc_std:.4f})')

    mlflow.log_metric('auc_test_acc_mean', auc_test_acc_mean)
    mlflow.log_metric('auc_test_acc_std', auc_test_acc_std)
    mlflow.log_metric('auc_test_micro_f1_mean', auc_test_micro_f1_mean)
    mlflow.log_metric('auc_test_micro_f1_std', auc_test_micro_f1_std)

    query_id = metrics_tracker.measured_metrics['query_id'].max()
    query_rows = metrics_tracker.measured_metrics[
        metrics_tracker.measured_metrics['query_id'] == query_id]

    test_acc_mean = query_rows['test_acc'].mean()
    test_acc_std = query_rows['test_acc'].std(ddof=0)
    test_micro_f1_mean = query_rows['test_micro_f1'].mean()
    test_micro_f1_std = query_rows['test_micro_f1'].std(ddof=0)

    mlflow.log_metric('test_acc_mean', test_acc_mean)
    mlflow.log_metric('test_acc_std', test_acc_std)
    mlflow.log_metric('test_micro_f1_mean', test_micro_f1_mean)
    mlflow.log_metric('test_micro_f1_std', test_micro_f1_std)

    logging.info(f'ACC: {test_acc_mean:.4f} (+/- {test_acc_std:.4f})')
    logging.info(f'mF1: {test_micro_f1_mean:.4f} (+/- {test_micro_f1_std:.4f})')
    logging.info('#--------------------------------')


def compute_auc_from_metrics_df(df_metrics):

    def f_agg(df):
        auc_x = df['num_samples'].tolist()
        span_x = (auc_x[-1] - auc_x[0])

        auc_y_acc = df['test_acc'].tolist()
        auc_y_f1 = df['test_micro_f1'].tolist()

        df_result = pd.DataFrame([
            [auc(auc_x, auc_y_acc) / span_x, auc(auc_x, auc_y_f1) / span_x]
        ], columns=['auc_test_acc', 'auc_test_micro_f1'])

        return df_result

    df_result = df_metrics.groupby('run_id').apply(f_agg)
    df_result.index = df_result.index.levels[0]

    return df_result


def get_initial_indices(train_set, initialization_strategy, num_samples):

    if initialization_strategy == 'random':
        from small_text.initialization import random_initialization
        indices_initial = random_initialization(train_set, n_samples=num_samples)
    elif initialization_strategy == 'srandom':
        from small_text.initialization import random_initialization_stratified
        y_train = train_set.y
        indices_initial = random_initialization_stratified(y_train, n_samples=num_samples)
    elif initialization_strategy == 'balanced':
        from small_text.initialization import random_initialization_balanced
        y_train = train_set.y
        indices_initial = random_initialization_balanced(y_train, n_samples=num_samples)
    else:
        raise ValueError('Invalid initialization strategy: ' + initialization_strategy)

    return indices_initial
