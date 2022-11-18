import math
import torch

import numpy as np

from enum import Enum

from sklearn.preprocessing import normalize
from small_text.data.datasets import SklearnDataset
from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
from small_text.integrations.transformers.datasets import TransformersDataset
from transformers import AutoTokenizer


TEST_SET_RATIO_DEFAULT = 0.1


class DataSets(Enum):
    AG_NEWS = 'ag-news'
    TREC = 'trec'
    MR = 'mr'
    SUBJ = 'subj'
    CR = 'CR'

    @staticmethod
    def from_str(enum_str):
        if enum_str == 'ag-news':
            return DataSets.AG_NEWS
        elif enum_str == 'trec':
            return DataSets.TREC
        elif enum_str == 'mr':
            return DataSets.MR
        elif enum_str == 'subj':
            return DataSets.SUBJ
        elif enum_str == 'cr':
            return DataSets.CR

        raise ValueError('Enum DataSets does not contain the given element: '
                         '\'{}\''.format(enum_str))


class DataSetType(Enum):
    TENSOR_PADDED_SEQ = 'tps'
    BOW = 'bow'
    RAW = 'raw'
    TRANSFORMERS = 'transformers'

    @staticmethod
    def from_str(enum_str):
        if enum_str == 'tps':
            return DataSetType.TENSOR_PADDED_SEQ
        elif enum_str == 'bow':
            return DataSetType.BOW
        elif enum_str == 'raw':
            return DataSetType.RAW
        elif enum_str == 'transformers':
            return DataSetType.TRANSFORMERS

        raise ValueError('Enum DataSetType does not contain the given element: '
                         '\'{}\''.format(enum_str))


class RawDataset(SklearnDataset):

    def __init__(self, x, y, target_labels=None):
        super().__init__(x, y, target_labels=target_labels)
        self.x = np.array(x)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
        if self.track_target_labels:
            self._infer_target_labels(self._y)

    @property
    def target_labels(self):
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        self._target_labels = target_labels

    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray) or isinstance(item, slice):
            return RawDataset(np.array(self._x[item]), np.array(self._y[item]))

        ds = RawDataset(self._x[item], self._y[item])
        if len(ds._x.shape) <= 1:
            ds._x = np.expand_dims(ds._x, axis=0)
            ds._y = np.expand_dims(ds._y, axis=0)

        return ds

    def __iter__(self):
        for i in range(self._x.shape[0]):
            yield self[i]

    def __len__(self):
        return self._x.shape[0]


class UnknownDataSetException(ValueError):
    pass


def load_dataset(dataset, dataset_kwargs, classifier_name, classifier_kwargs, dataset_type=None):
    """

    Parameters
    ----------
    dataset : str
        Dataset name which can be consumed by `DataSets.from_str()`.
    dataset_kwargs : dict
        Additional arguments for the dataset.
    classifier_name : str
        Name of the classifier. Can be used to infer the dataset type.
    classifier_kwargs : dict
        Additional arguments for the classifier. Some arguments from this may be used for
        constructing a transformers tokenizer.
    dataset_type : DataSetType, default=None
        Data set type or None.

    Returns
    -------
    train : Dataset
        Training set for the dataset identified by `dataset_name`.
    test : Dataset
        Test set for the dataset identified by `dataset_name`.
    """
    dataset_type_expected = get_dataset_type(classifier_name, dataset_kwargs, dataset_type)
    dataset = DataSets.from_str(dataset)

    train, test = _load_dataset(dataset, dataset_type_expected, dataset_kwargs, classifier_kwargs)

    return train, test


def get_dataset_type(classifier_name, dataset_kwargs, dataset_type):

    if 'dataset_type' in dataset_kwargs:
        dataset_type_expected = DataSetType.from_str(dataset_kwargs['dataset_type'])
    elif dataset_type is not None:
        if isinstance(dataset_type, DataSetType):
            dataset_type_expected = dataset_type
        else:
            dataset_type_expected = DataSetType.from_str(dataset_type)
    else:
        dataset_type_expected = get_dataset_type_for_classifier(classifier_name)

    return dataset_type_expected


def get_dataset_type_for_classifier(classifier_name):

    if classifier_name == 'svm':
        return DataSetType.BOW
    elif classifier_name == 'kimcnn':
        return DataSetType.TENSOR_PADDED_SEQ
    elif classifier_name == 'transformer':
        return DataSetType.TRANSFORMERS

    raise ValueError(f'No dataset type defined for classifier_name {classifier_name}')


def _raise_invalid_format_error():
    raise ValueError('Invalid dataset format: '
                     'Dataset description string must be in the format '
                     '\'[DATASET]--[DATASET-TYPE]\'.')


def _load_dataset(dataset, dataset_type, dataset_kwargs, classifier_kwargs,
                  test_set_ratio=TEST_SET_RATIO_DEFAULT):

    if dataset == DataSets.AG_NEWS:
        return _load_agn(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.TREC:
        return _load_trec(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.MR:
        return _load_mr(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.CR:
        return _load_cr(dataset, dataset_type, dataset_kwargs, classifier_kwargs,
                        test_set_ratio=test_set_ratio)
    elif dataset == DataSets.SUBJ:
        return _load_subj(dataset, dataset_type, dataset_kwargs, classifier_kwargs,
                          test_set_ratio=test_set_ratio)

    raise UnknownDataSetException(f'Unknown dataset / type combination: '
                                  f'{dataset} - {str(dataset_type)}')


def _load_agn(dataset, dataset_type, dataset_kwargs, classifier_kwargs):

    import datasets
    agn_dataset = datasets.load_dataset('ag_news')

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             agn_dataset['train']['text'],
                                             agn_dataset['train']['label'],
                                             agn_dataset['test']['text'],
                                             agn_dataset['test']['label'],
                                             dataset_kwargs['max_length'])
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            agn_dataset['train']['text'],
            agn_dataset['train']['label'],
            agn_dataset['test']['text'],
            agn_dataset['test']['label'])
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(agn_dataset['train']['text'],
                            agn_dataset['train']['label'],
                            agn_dataset['test']['text'],
                            agn_dataset['test']['label'])
    elif dataset_type == DataSetType.RAW:
        return RawDataset(agn_dataset['train']['text'],
                          agn_dataset['train']['label']), \
               RawDataset(agn_dataset['test']['text'],
                          agn_dataset['test']['label'])
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_trec(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    trec_dataset = datasets.load_dataset('trec')

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             trec_dataset['train']['text'],
                                             trec_dataset['train']['label-coarse'],
                                             trec_dataset['test']['text'],
                                             trec_dataset['test']['label-coarse'],
                                             dataset_kwargs['max_length'])
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            trec_dataset['train']['text'],
            trec_dataset['train']['label-coarse'],
            trec_dataset['test']['text'],
            trec_dataset['test']['label-coarse'])
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(trec_dataset['train']['text'],
                            trec_dataset['train']['label-coarse'],
                            trec_dataset['test']['text'],
                            trec_dataset['test']['label-coarse'])
    elif dataset_type == DataSetType.RAW:
        return RawDataset(trec_dataset['train']['text'],
                          trec_dataset['train']['label-coarse']), \
               RawDataset(trec_dataset['test']['text'],
                          trec_dataset['test']['label-coarse'])
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_mr(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    mr_dataset = datasets.load_dataset('rotten_tomatoes')

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(dataset_kwargs, classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             mr_dataset['train']['text'] + mr_dataset['validation'][
                                                 'text'],
                                             mr_dataset['train']['label'] +
                                             mr_dataset['validation'][
                                                 'label'],
                                             mr_dataset['test']['text'],
                                             mr_dataset['test']['label'],
                                             int(dataset_kwargs['max_length']))
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            mr_dataset['train']['text'], mr_dataset['train']['label'],
            mr_dataset['test']['text'], mr_dataset['test']['label'])
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(mr_dataset['train']['text'],
                            mr_dataset['train']['label'],
                            mr_dataset['test']['text'],
                            mr_dataset['test']['label'])
    elif dataset_type == DataSetType.RAW:
        return RawDataset(mr_dataset['train']['text'] + mr_dataset['validation']['text'],
                          mr_dataset['train']['label'] + mr_dataset['validation']['label']), \
               RawDataset(mr_dataset['test']['text'],
                          mr_dataset['test']['label'])
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_cr(dataset, dataset_type, dataset_kwargs, classifier_kwargs, test_set_ratio=0.1):
    import gluonnlp
    cr = gluonnlp.data.CR()

    test_set_size = int(math.ceil(len(cr) * test_set_ratio))
    indices = np.random.permutation(len(cr))

    train = [cr[i] for i in indices[test_set_size:]]
    test = [cr[i] for i in indices[:test_set_size]]

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)

        return _text_to_transformers_dataset(tokenizer,
                                             [item[0] for item in train],
                                             [item[1] for item in train],
                                             [item[0] for item in test],
                                             [item[1] for item in test],
                                             dataset_kwargs['max_length'])
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            [item[0] for item in train],
            [item[1] for item in train],
            [item[0] for item in test],
            [item[1] for item in test])
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow([item[0] for item in train],
                            [item[1] for item in train],
                            [item[0] for item in test],
                            [item[1] for item in test])
    elif dataset_type == DataSetType.RAW:
        return RawDataset([item[0] for item in train],
                          [item[1] for item in train]), \
               RawDataset([item[0] for item in test],
                          [item[1] for item in test])
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_subj(dataset, dataset_type, dataset_kwargs, classifier_kwargs, test_set_ratio=0.1):
    import gluonnlp
    subj = gluonnlp.data.SUBJ()

    test_set_size = int(math.ceil(len(subj) * test_set_ratio))
    indices = np.random.permutation(len(subj))

    train = [subj[i] for i in indices[test_set_size:]]
    test = [subj[i] for i in indices[:test_set_size]]

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)

        return _text_to_transformers_dataset(tokenizer,
                                             [item[0] for item in train],
                                             [item[1] for item in train],
                                             [item[0] for item in test],
                                             [item[1] for item in test],
                                             dataset_kwargs['max_length'])
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            [item[0] for item in train],
            [item[1] for item in train],
            [item[0] for item in test],
            [item[1] for item in test])
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow([item[0] for item in train],
                            [item[1] for item in train],
                            [item[0] for item in test],
                            [item[1] for item in test])
    elif dataset_type == DataSetType.RAW:
        return RawDataset([item[0] for item in train],
                          [item[1] for item in train]), \
               RawDataset([item[0] for item in test],
                          [item[1] for item in test])
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _get_huggingface_tokenizer(classifier_kwargs):
    tokenizer_name = classifier_kwargs['transformer_tokenizer'] \
        if 'transformer_tokenizer' in classifier_kwargs else classifier_kwargs['transformer_model']
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir='.transformers_cache/',
    )
    return tokenizer


def _create_bow_preprocessor(preprocessor):
    result = preprocessor.vectorize((1, 2), preprocessor.get_train_docs(),
                                    preprocessor.get_test_docs())

    return (SklearnDataset(normalize(result['x']), result['y']),
            SklearnDataset(normalize(result['x_test']), result['y_test']))


def _text_to_bow(x, y, x_test, y_test, max_features=50000, ngram_range=(1, 2)):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    x = vectorizer.fit_transform(x)
    x_test = vectorizer.transform(x_test)

    return (SklearnDataset(normalize(x), y),
            SklearnDataset(normalize(x_test), y_test))


def _text_to_transformers_dataset(tokenizer, train_text, train_labels, test_text,
                                  test_labels, max_length):

    return _transformers_prepare(tokenizer, train_text, train_labels, max_length=max_length), \
           _transformers_prepare(tokenizer, test_text, test_labels, max_length=max_length)


def _transformers_prepare(tokenizer, data, labels, max_length=60):

    data_out = []
    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )

        data_out.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i]))

    return TransformersDataset(data_out)


def _text_to_text_classification_dataset(x_text, y, x_test_text, y_test):
    try:
        from torchtext.legacy import data
    except AttributeError:
        from torchtext import data

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False, unk_token=None, pad_token=None)

    fields = [('text', text_field), ('label', label_field)]

    train = data.Dataset([data.Example.fromlist([text, labels], fields)
                          for text, labels in zip(x_text, y)],
                         fields)
    test = data.Dataset([data.Example.fromlist([text, labels], fields)
                         for text, labels in zip(x_test_text, y_test)],
                        fields)

    text_field.build_vocab(train, min_freq=1)
    label_field.build_vocab(train)

    train_tc = _tt_dataset_to_text_classification_dataset(train)
    test_tc = _tt_dataset_to_text_classification_dataset(test)

    return train_tc, test_tc


def _tt_dataset_to_text_classification_dataset(dataset):
    assert dataset.fields['text'].vocab.itos[0] == '<unk>'
    assert dataset.fields['text'].vocab.itos[1] == '<pad>'
    unk_token_idx = 1

    vocab = dataset.fields['text'].vocab

    data = [
        (torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                           for token in example.text]),
         dataset.fields['label'].vocab.stoi[example.label])
        for example in dataset.examples
    ]

    return PytorchTextClassificationDataset(data, vocab)
