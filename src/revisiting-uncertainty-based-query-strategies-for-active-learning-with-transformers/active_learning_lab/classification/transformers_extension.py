import tempfile
import types

import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler

from transformers import get_linear_schedule_with_warmup

from small_text.integrations.transformers.classifiers.classification import _get_layer_params
from small_text.integrations.transformers.classifiers.factories import (
    TransformerBasedClassificationFactory
)
from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.utils.system import get_tmp_dir_base


# suppresses an unnecessary log warning that was shown in small-text v1.0.0a8
def _fit_main(self, sub_train, sub_valid, optimizer, scheduler):
    if self.model is None:
        y = [entry[TransformersDataset.INDEX_LABEL] for entry in sub_train]
        if self.num_classes is None:
            self.num_classes = np.max(y) + 1

        if self.num_classes != np.max(y) + 1:
            raise ValueError('Conflicting information about the number of classes: '
                             'expected: {}, encountered: {}'.format(self.num_classes,
                                                                    np.max(y) + 1))

        self.initialize_transformer(self.cache_dir)

    if self.criterion is None:
        self.criterion = self.get_default_criterion()

    if self.fine_tuning_arguments is not None:
        params = _get_layer_params(self.model, self.lr, self.fine_tuning_arguments)
    else:
        params = None

    if optimizer is None or scheduler is None:

        optimizer, scheduler = self._initialize_optimizer_and_scheduler(optimizer,
                                                                        scheduler,
                                                                        self.fine_tuning_arguments,
                                                                        self.lr,
                                                                        params,
                                                                        self.model,
                                                                        sub_train)

    self.model = self.model.to(self.device)

    with tempfile.TemporaryDirectory(dir=get_tmp_dir_base()) as tmp_dir:
        res = self._train(sub_train, sub_valid, tmp_dir, optimizer, scheduler)
        self._perform_model_selection(sub_valid)

    return res


def _initialize_optimizer_and_scheduler(self, optimizer, scheduler, fine_tuning_arguments,
                                        base_lr, _params, model, sub_train):
    steps = (len(sub_train) // self.mini_batch_size) \
            + int(len(sub_train) % self.mini_batch_size != 0)

    params = _get_params(model, base_lr, fine_tuning_arguments)

    optimizer = self._default_optimizer(params, base_lr) if optimizer is None else optimizer
    total_steps = steps * self.num_epochs

    if scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=steps * self.num_epochs)
    elif scheduler == 'slanted':
        optimizer = AdamW(params, lr=base_lr, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0.1*total_steps,
                                                    num_training_steps=steps*self.num_epochs)

    elif not isinstance(scheduler, _LRScheduler):
        raise ValueError(f'Invalid scheduler: {scheduler}')

    return optimizer, scheduler


def _get_params(model, base_lr, fine_tuning_arguments):

    base_model = getattr(model, model.base_model_prefix)
    num_layers = len(base_model.encoder.layer)
    layer_prefix = f'{model.base_model_prefix}.encoder.layer'
    groups = [(f'{layer_prefix}.{i}.', base_lr) for i in range(num_layers)]

    params = []

    for i, (l, lr) in enumerate(groups):
        params.extend(
            [
                {
                    'params': [p for n, p in model.named_parameters() if
                               'bias' not in n and n.startswith(l)],
                    'weight_decay_rate': 0.01,
                    'lr': lr * fine_tuning_arguments.layerwise_gradient_decay ** (
                                num_layers - i - 1)
                },
                {
                    'params': [p for n, p in model.named_parameters() if
                               'bias' in n and n.startswith(l)],
                    'weight_decay_rate': 0.0,
                    'lr': lr * fine_tuning_arguments.layerwise_gradient_decay ** (
                                num_layers - i - 1)
                },
            ]
        )

    params.extend([
        {
            'params': [p for n, p in model.named_parameters()
                       if 'bias' not in n and not n.startswith(f'{layer_prefix}.')],
            'weight_decay_rate': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if 'bias' in n and not n.startswith(f'{layer_prefix}.')],
            'weight_decay_rate': 0.0
        },
    ])

    return params


class TransformerBasedClassificationExtendedFactory(TransformerBasedClassificationFactory):

    def new(self):
        clf = super().new()
        clf._initialize_optimizer_and_scheduler = types.MethodType(
            _initialize_optimizer_and_scheduler, clf)
        clf._fit_main = types.MethodType(_fit_main, clf)

        return clf
