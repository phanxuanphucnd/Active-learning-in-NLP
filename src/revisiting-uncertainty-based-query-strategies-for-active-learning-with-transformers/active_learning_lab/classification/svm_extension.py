from small_text.classifiers.classification import (
    EmbeddingMixin,
    SklearnClassifier
)
from small_text.classifiers.factories import SklearnClassifierFactory


class SklearnClassifierWithEmbeddings(EmbeddingMixin, SklearnClassifier):

    def embed(self, dataset, return_proba=True, pbar=None, embedding_method=None):
        """These "embeddings" are just the sparse vectors that are used as embedding vectors.
        They are only used for kNN search in Contrastive Active Learning."""
        if return_proba:
            return dataset.x, self.predict_proba(dataset)
        return dataset.x


class SklearnClassifierFactoryExtended(SklearnClassifierFactory):

    def new(self):
        return SklearnClassifierWithEmbeddings(self.base_estimator_class(**self.kwargs))
