import numpy as np
from scipy.special import rel_entr

from small_text.query_strategies import (
    BreakingTies,
    EmbeddingBasedQueryStrategy,
    LeastConfidence,
    RandomSampling,
    PredictionEntropy,
    SubsamplingQueryStrategy)


def query_strategy_from_str(query_strategy_name, kwargs):

    if query_strategy_name == 'bt':
        strategy = BreakingTies()
    elif query_strategy_name == 'lc':
        strategy = LeastConfidence()
    elif query_strategy_name == 'pe':
        strategy = PredictionEntropy()
    elif query_strategy_name == 'ca':
        strategy = ContrastiveActiveLearning()
    elif query_strategy_name == 'rd':
        strategy = RandomSampling()
    else:
        raise ValueError(f'Unknown query strategy string: {query_strategy_name}')

    if kwargs is not None and 'subsample' in kwargs:
        subsample_size = int(kwargs['subsample'])
        strategy = SubsamplingQueryStrategy(strategy, subsample_size)

    return strategy


class ContrastiveActiveLearning(EmbeddingBasedQueryStrategy):
    """Contrastive Active Learning [MVB+21]_ selects instances whose k-nearest neighbours
    exhibit the largest mean Kullback-Leibler divergence.

    .. [MVB+21] Katerina Margatina, Giorgos Vernikos, Loïc Barrault, and Nikolaos Aletras. 2021.
       Active Learning by Acquiring Contrastive Examples.
       In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing,
          pages 650–663.
    """

    def __init__(self, k=10, embed_kwargs=dict(), normalize=True, batch_size=100, pbar='tqdm'):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbours whose KL divergence is considered.
        embed_kwargs : dict
            Embedding keyword args which are passed to `clf.embed()`.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        batch_size : int, default=100
            Batch size which is used to process the embeddings.
        """
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize
        self.k = k
        self.batch_size = batch_size
        self.pbar = pbar

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm',
              embeddings=None, embed_kwargs=dict()):

        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n,
                             embed_kwargs=self.embed_kwargs, pbar=self.pbar)

    def sample(self, _clf, dataset, indices_unlabeled, _indices_labeled, _y, n, embeddings,
               embeddings_proba=None):
        from sklearn.neighbors import NearestNeighbors

        if embeddings_proba is None:
            raise ValueError('Error: embeddings_proba is None. '
                             'This strategy requires a classifier whose embed() method '
                             'supports the return_proba kwarg.')

        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        nn = NearestNeighbors(n_neighbors=n)
        nn.fit(embeddings)

        return self._contrastive_active_learning(dataset, embeddings, embeddings_proba,
                                                 indices_unlabeled, nn, n)

    def _contrastive_active_learning(self, dataset, embeddings, embeddings_proba,
                                     indices_unlabeled, nn, n):
        scores = []

        embeddings_unlabelled_proba = embeddings_proba[indices_unlabeled]
        embeddings_unlabeled = embeddings[indices_unlabeled]

        num_batches = int(np.ceil(len(dataset) / self.batch_size))
        offset = 0
        for batch_idx in np.array_split(np.arange(indices_unlabeled.shape[0]), num_batches,
                                        axis=0):

            nn_indices = nn.kneighbors(embeddings_unlabeled[batch_idx],
                                       n_neighbors=self.k,
                                       return_distance=False)

            kl_divs = np.apply_along_axis(lambda v: np.mean([
                rel_entr(embeddings_proba[i], embeddings_unlabelled_proba[v])
                for i in nn_indices[v - offset]]),
                0,
                batch_idx[None, :])

            scores.extend(kl_divs.tolist())
            offset += batch_idx.shape[0]

        scores = np.array(scores)
        indices = np.argpartition(-scores, n)[:n]

        return indices

    def __str__(self):
        return f'ContrastiveActiveLearning(k={self.k}, ' \
               f'embed_kwargs={str(self.embed_kwargs)}, ' \
               f'normalize={self.normalize})'
