import torch
import numpy as np

from pathlib import Path


def get_embedding_matrix(name, vocab, data_dir='.data/'):
    from gensim.models.word2vec import Word2VecKeyedVectors

    embedding_dir = Path(data_dir).joinpath('embeddings')
    embedding_dir.mkdir(parents=True, exist_ok=True)

    serialized_file = embedding_dir.joinpath(name + '.bin')
    if not serialized_file.exists():
        import gensim.downloader as api
        model = api.load('word2vec-google-news-300')
        model.save(str(serialized_file.resolve()))
        return _build_embedding_matrix_from_keyedvectors(model, vocab)
    else:
        model = Word2VecKeyedVectors.load(str(serialized_file.resolve()), mmap='r')
        return _build_embedding_matrix_from_keyedvectors(model, vocab)


def _build_embedding_matrix_from_keyedvectors(pretrained_vectors, vocab, min_freq=1):
    vectors = [
        np.zeros(pretrained_vectors.vectors.shape[1])  # <pad>
    ]
    num_special_vectors = len(vectors)
    vectors += [
        pretrained_vectors.vectors[pretrained_vectors.vocab[vocab.itos[i]].index]
        if vocab.itos[i] in pretrained_vectors.vocab
        else np.zeros(pretrained_vectors.vectors.shape[1])
        for i in range(num_special_vectors, len(vocab))
    ]
    for i in range(num_special_vectors, len(vocab)):
        if vocab.itos[i] not in pretrained_vectors.vocab and vocab.freqs[vocab.itos[i]] >= min_freq:
            vectors[i] = np.random.uniform(-0.25, 0.25, pretrained_vectors.vectors.shape[1])

    return torch.as_tensor(np.stack(vectors))
