import types
import torch

import numpy as np
import torch.nn.functional as F

from small_text.utils.context import build_pbar_context
from small_text.utils.data import list_length
from small_text.integrations.pytorch.utils.data import dataloader
from small_text.integrations.pytorch.classifiers import KimCNNFactory


# creates pooled representations
def embed(self, data_set, return_proba=False, _embedding_method='unused',
          _module_selector=lambda x: x['fc'], pbar='tqdm'):
    if self.model is None:
        raise ValueError('Model is not trained. Please call fit() first.')

    self.model.eval()

    dataset_iter = dataloader(data_set.data, self.mini_batch_size, self._create_collate_fn(),
                              train=False)

    tensors = []
    proba = []
    with build_pbar_context(pbar, tqdm_kwargs={'total': list_length(data_set)}) as pbar:
        for text, _ in dataset_iter:
            text = text.to(self.device)
            batch_len = text.size(0)

            text = self.model.embedding(text)
            text = text.unsqueeze(dim=1)

            out_tensors = []
            for conv, pool in zip(self.model.convs, self.model.pools):
                activation = pool(F.relu(conv(text)))
                out_tensors.append(activation)

            text = torch.cat(out_tensors, dim=1)

            batch_size = text.size(0)
            text = text.view(batch_size, -1)

            tensors.extend(text.detach().to('cpu', non_blocking=True).numpy())

            if return_proba:
                proba.extend(F.softmax(self.model.fc(text), dim=1).detach().to('cpu').tolist())

            pbar.update(batch_len)

    if return_proba:
        return np.array(tensors), np.array(proba)

    return np.array(tensors)


# this is a fix because model.eval() was missing here in small-text==1.0.0a8
def validate(self, validation_set):
    self.model.eval()

    valid_loss = 0.
    acc = 0.

    valid_iter = dataloader(validation_set.data, self.mini_batch_size, self._create_collate_fn(),
                            train=False)

    for x, cls in valid_iter:
        x, cls = x.to(self.device), cls.to(self.device)

        with torch.no_grad():
            output = self.model(x)
            loss = self.criterion(output, cls)
            valid_loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()
            del output, x, cls

    return valid_loss / len(validation_set), acc / len(validation_set)


class KimCNNExtendedFactory(KimCNNFactory):

    def new(self):
        clf = super().new()

        clf.embed = types.MethodType(embed, clf)
        clf.validate = types.MethodType(validate, clf)

        return clf
