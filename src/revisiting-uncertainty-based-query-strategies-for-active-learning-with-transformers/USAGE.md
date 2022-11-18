# Usage Instructions

The entry point for this script is `active_learning_lab/run_experiment.py`.
It is recommended to run this experiment setup as a module (i.e. with `python -m`). 
Passing `-h` will output the help with the full set of parameters:

```
python -m active_learning_lab.run_experiment -h
```

**Output**:
```
usage: run_experiment.py [-h] [--description DESCRIPTION] [--runs RUNS]
                         [--seed SEED] [--max-reproducibility]
                         [--dataset_name DATASET_NAME]
                         [--dataset_kwargs DATASET_KWARGS]
                         [--classifier_name CLASSIFIER_NAME]
                         [--validation_set_size VALIDATION_SET_SIZE]
                         [--classifier_kwargs CLASSIFIER_KWARGS]
                         [--num_queries NUM_QUERIES] [--query_size QUERY_SIZE]
                         [--initialization_strategy INITIALIZATION_STRATEGY]
                         [--initialization_strategy_kwargs INITIALIZATION_STRATEGY_KWARGS]
                         [--query_strategy QUERY_STRATEGY]
                         [--query_strategy_kwargs QUERY_STRATEGY_KWARGS]
                         experiment_name config_files [config_files ...]

Runs an active learning experiment.

[... here you will see a description for each parameter ...]
[... details omitted in USAGE.md for brevity's sake ...]

Process finished with exit code 0

```

In order to handle these numerous options, the argument parser has been extended 
to handle a list of config files with an override mechanism where the last config wins. 
The direct use of kwargs has lower precedence than the same setting within a config file,
but can be used to explore the parameters.

The most important settings are classifier name, dataset, and query strategy.
The possible values are listed below:

- `classifier_name`: svm, kimcnn, distilroberta, bert
- `dataset`: ag-news, cr, mr, subj, trec
- `query_strategy`: pe, bt, lc, ca, rd

## Before the first run

Before the first run you must set up an mlflow experiment.

Example:

```
mlflow experiments create -n active-learning
```

where `active-learning` is the name of your respective mlflow experiment. 
The experiment name is arbitrary but must match the name passed to `run_experiment.py`.

## Run Helper Script

To demonstrate the use of aforementioned config files we have a run helper (bash) script, 
called [run.sh](run.sh), which can be used as follows:

```bash
run.sh [experiment_name] [classifier_name] [dataset_name] [query_strategy]
```

where `experiment_name` refers to the mlflow experiment and `classifier_name`, `dataset_name`, 
and `query_strategy` are each one of the values listed before.

Our config files can be found in the [config/](config/) folder.

## Inspecting the Results

All result tracking is handled by mlflow which writes all experiments to a subfolder which is located at:

```<working directory>/mlruns/<experiment id>/<experiment folder>```

where `experiment id` can be obtained via the command `mlflow experiments list` and 
`experiment folder` is an unique alphanumeric string (e.g., `5ee72ff3bc6f46e49cd601e9eec1a585`) 
for each run. Such a folder corresponds to an mlflow run and contains the results of a single experiment's execution.

Within such a folder you will the following files and folders:
```bash
> ls -F
artifacts/  meta.yaml  metrics/  params/  tags/
```

All tracked results will be saved under `artifacts/`:
```bash
> ls -F
auc.csv  out.log  queries.npz  results_agg.csv	results.csv  run_1/ run_2/ run_3/ run_4/ run_5/  test_labels.npz  train_labels.npz
```

Most results are either text files (.log, .txt, .csv) or serialized numpy files (.npz). 
Notable files are:

- `artifacts/auc.csv`: contains the AUC scores per single run
- `artifacts/results.csv`: contains the result of every single query
- `artifacts/results_agg.csv`: contains the results, aggregated per query step into mean and stddev
