import torch
import numpy as np

from active_learning_lab.classification.factories import get_factory
from active_learning_lab.data.data import load_dataset

from active_learning_lab.experiments.experiment import ActiveLearningExperiment, \
    ClassificationConfig, ExperimentConfig, DatasetConfig
from active_learning_lab.utils.pytorch import default_tensor_type


class ActiveLearningExperimentBuilder(object):

    def __init__(self, num_queries, query_size, query_strategy, query_strategy_kwargs, cv, tmp_dir):
        self.num_queries = num_queries
        self.query_size = query_size
        self.query_strategy = query_strategy
        self.query_strategy_kwargs = query_strategy_kwargs

        self.cv = cv
        self.tmp_dir = tmp_dir

        self.classifier_name = None
        self.classifier_kwargs = None
        self.classifier_factory = None
        self.train = None
        self.test = None
        self.initialization_strategy = None
        self.initialization_strategy_kwargs = None

    def with_dataset(self, dataset_name, dataset_kwargs):

        if self.classifier_name is None:
            raise ValueError('classifier_name must be set prior to assigning the dataset_name')

        if dataset_kwargs is None:
            dataset_kwargs = dict()

        if self.classifier_name == 'transformer':
            if 'transformer_model' not in self.classifier_kwargs:
                raise RuntimeError('Key \'transformer_model\' not set in transformer_model. '
                                   'This should not happen.')

            dataset_kwargs['tokenizer_name'] = self.classifier_kwargs['transformer_model']

        train_raw, test_raw = load_dataset(dataset_name,
                                           dataset_kwargs,
                                           self.classifier_name,
                                           self.classifier_kwargs,
                                           dataset_type='raw')

        with default_tensor_type(torch.FloatTensor):
            self.train, self.test = load_dataset(dataset_name,
                                                 dataset_kwargs,
                                                 self.classifier_name,
                                                 self.classifier_kwargs)
            self.num_classes = np.unique(self.train.y).shape[0]

        self.dataset_config = DatasetConfig(dataset_name, dataset_kwargs, train_raw, test_raw)

        return self

    def with_classifier(self, classifier_name, validation_set_size, classifier_kwargs, classifier_factory=None):

        self.classifier_name = classifier_name
        self.validation_set_size = validation_set_size

        self.classifier_kwargs = dict() if classifier_kwargs is None else classifier_kwargs

        self.classifier_factory = classifier_factory

        self.incremental_training = self.classifier_kwargs.get('incremental_training', False)
        if 'incremental_training' in self.classifier_kwargs:
            del self.classifier_kwargs['incremental_training']

        if self.classifier_name == 'transformer':
            if 'transformer_model' not in self.classifier_kwargs:
                raise ValueError('\'transformer_model\' not set in classifier_kwargs')

        return self

    def with_initialization(self, initialization_strategy, initialization_strategy_kwargs):
        self.initialization_strategy = initialization_strategy
        if initialization_strategy_kwargs is None:
            self.initialization_strategy_kwargs = dict()
        else:
            self.initialization_strategy_kwargs = initialization_strategy_kwargs
        return self

    def build(self):
        exp_args = ExperimentConfig(self.cv, self.num_queries, self.query_size)

        if self.classifier_factory is None:

            if self.classifier_name == 'kimcnn':
                from active_learning_lab.data.embeddings import get_embedding_matrix
                self.classifier_kwargs['embedding_matrix'] = get_embedding_matrix(
                    self.classifier_kwargs['embedding_matrix'],
                    self.train.vocab)

            self.classifier_factory = get_factory(self.classifier_name,
                                                  self.num_classes,
                                                  classifier_kwargs=self.classifier_kwargs)

        classification_config = ClassificationConfig(self.classifier_name,
                                                     self.classifier_factory,
                                                     classifier_kwargs=self.classifier_kwargs,
                                                     incremental_training=self.incremental_training,
                                                     validation_set_size=self.validation_set_size)

        return ActiveLearningExperiment(exp_args,
                                        classification_config,
                                        self.dataset_config,
                                        self.initialization_strategy,
                                        self.initialization_strategy_kwargs,
                                        self.train, self.tmp_dir,
                                        query_strategy=self.query_strategy,
                                        query_strategy_kwargs=self.query_strategy_kwargs)
