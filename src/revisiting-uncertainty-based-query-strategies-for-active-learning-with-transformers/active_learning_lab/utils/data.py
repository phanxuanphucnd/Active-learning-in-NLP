from small_text.utils.data import list_length
from small_text.data.sampling import stratified_sampling, balanced_sampling


def get_validation_set(y, strategy='balanced', validation_set_size=0.1):

    if validation_set_size == 0.0:
        return None

    n_samples = int(validation_set_size * list_length(y))

    if strategy == 'balanced':
        return balanced_sampling(y, n_samples=n_samples)
    elif strategy == 'stratified':
        return stratified_sampling(y, n_samples=n_samples)

    raise ValueError(f'Invalid strategy: {strategy}')
