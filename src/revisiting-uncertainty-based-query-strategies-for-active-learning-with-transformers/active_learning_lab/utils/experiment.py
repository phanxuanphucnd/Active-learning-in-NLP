import os
import torch
import warnings

import numpy as np

from pathlib import Path


def set_random_seed(seed, pytorch=True):
    # PYTHONHASHSEED and numpy seed have the smaller range (2**32-1)
    assert 0 <= seed <= 2**32-1

    os.environ['PYTHONHASHSEED'] = str(seed)
    if pytorch:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def write_experiment_info(logger, mlflow_obj, name, sep_width=80):
    active_run = mlflow_obj.active_run()

    logger.info('*')
    logger.info('-' * sep_width)
    logger.info('Experiment: %s (ID: %s)', name, active_run.info.experiment_id)
    logger.info('Active run: %s', active_run.info.run_id)
    logger.info('-' * sep_width)


def get_tmp_path(run):
    """Returns a tmp dir relative to the given mlflow active run.

    Parameters
    ----------
    run : mlflow.ActiveRun
        an mlflow.ActiveRun object

    Returns
    -------
    path : Path
        path to a tmp directory relative to the current run directory
    """

    base_path = run.info.artifact_uri
    if ':' in base_path:
        base_path = base_path[run.info.artifact_uri.index(':') + 1:]
    base_path = Path(base_path).joinpath('..')
    tmp_path = base_path.joinpath('tmp/').resolve()
    tmp_path.resolve().mkdir()

    return tmp_path


def free_resources_fix():
    import gc
    # this should not be necessary, but especially GPU tensors seemed to be released earlier
    # when explicitly calling this
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def suppress_known_thirdparty_warnings():
    warnings.filterwarnings(action='ignore',
                            category=DeprecationWarning,
                            module=r'gensim.matutils')
    warnings.filterwarnings(action='ignore',
                            category=DeprecationWarning,
                            module=r'gensim.corpora.dictionary')
    warnings.filterwarnings(action='ignore',
                            category=DeprecationWarning,
                            module=r'gensim.models.doc2vec')
    warnings.filterwarnings(action='ignore',
                            category=DeprecationWarning,
                            module=r'mxnet.numpy.utils')
    # we could fix this but want to keep the pinned small-text version
    warnings.filterwarnings(action='ignore',
                            category=FutureWarning,
                            module=r'small_text.integrations.pytorch.utils.data')
    warnings.filterwarnings(action='ignore',
                            category=RuntimeWarning,
                            module=r'small_text.integrations.transformers.classifiers.classification')
