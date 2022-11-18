import re
import sys

import mlflow

from contextlib import closing
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

from IPython.utils.io import Tee

from active_learning_lab.args import get_parser
from active_learning_lab.experiments.experiment_builder import (
    ActiveLearningExperimentBuilder
)
from active_learning_lab.utils.experiment import (
    get_tmp_path,
    set_random_seed,
    suppress_known_thirdparty_warnings,
    write_experiment_info
)
from active_learning_lab.utils.logging import setup_logger


def main(args, experiment_name):

    sys.stderr = sys.stdout
    active_run = mlflow.active_run()

    tmp_path = get_tmp_path(active_run)
    log_file_path = tmp_path.joinpath('out.log').resolve()

    try:
        with TemporaryDirectory(dir=str(tmp_path.resolve())) as tmp_dir:
            with closing(Tee(str(log_file_path), 'w', channel='stdout')):
                logger = setup_logger()
                logger.info("Args: " + str(args))
                write_experiment_info(logger, mlflow, experiment_name)

                builder = get_active_learner_builder(args, tmp_dir)
                exp = builder.build()

                # [!] This is the entry point to the actual experiment
                results = exp.run(builder.train, builder.test)

                process_results(exp, results)

                write_experiment_info(logger, mlflow, experiment_name)
    finally:
        mlflow.log_artifact(str(log_file_path))
        rmtree(tmp_path, 'out.log')


def get_active_learner_builder(args, tmp_dir):

    builder = ActiveLearningExperimentBuilder(args['active_learner']['num_queries'],
                                              args['active_learner']['query_size'],
                                              args['active_learner']['query_strategy'],
                                              args['active_learner']['query_strategy_kwargs'],
                                              args['general']['runs'],
                                              str(tmp_dir)) \
        .with_classifier(args['classifier']['classifier_name'],
                         args['classifier']['validation_set_size'],
                         args['classifier']['classifier_kwargs']) \
        .with_initialization(args['active_learner']['initialization_strategy'],
                             args['active_learner']['initialization_strategy_kwargs']) \
        .with_dataset(args['dataset']['dataset_name'], args['dataset']['dataset_kwargs'])

    return builder


def process_results(al_exp, results):
    for name, file in results.artifacts:
        if '/' in name:
            artifact_dir = re.sub('^file:', '', mlflow.get_artifact_uri())
            basedir = Path(artifact_dir).joinpath(name).parents[0]
            if not basedir.exists():
                basedir.mkdir()
            file_rel = str(file.relative_to(al_exp.tmp_dir))
            mlflow.log_artifact(file, file_rel[:file_rel.rindex('/')])
        else:
            mlflow.log_artifact(file)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    suppress_known_thirdparty_warnings()
    set_random_seed(args['general']['seed'], args['general']['max_reproducibility'])

    client = mlflow.tracking.MlflowClient()

    experiment_name = args['experiment_name']
    experiment = mlflow.get_experiment_by_name(experiment_name)\

    if experiment is None:
        raise ValueError('No mlflow experiments with name \'{}\' exists. '
                         'Please create the experiment first.'.format(experiment_name))

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_param('experiment_name', experiment_name)
        mlflow.log_param('classifier_name', args['classifier']['classifier_name'])
        classifier_pretrained_model = args['classifier']['classifier_kwargs']['transformer_model'] \
            if args['classifier']['classifier_name'] == 'transformer' else ''
        mlflow.log_param('classifier_pretrained_model', classifier_pretrained_model)
        mlflow.log_param('dataset_name', args['dataset']['dataset_name'])
        mlflow.log_param('query_strategy', args['active_learner']['query_strategy'])
        mlflow.log_param('description', args['general']['description'])

        main(args, experiment_name)
