# USAGE: run.sh [experiment_name] [classifier_name] [dataset_name] [query_strategy]
#
# first argument: experiment_name ($1)
# second argument: classifier_name ($2)
# third argument: dataset_name ($3)
# fourth argument: query_strategy ($4)

SCRIPT_DIR=$(cd `dirname $0` && pwd)

EXPERIMENT_NAME=$1
CLASSIFIER_NAME=$2
DATASET=$3
QUERY_STRATEGY=$4

CONFIG_BASE_DIR=config

BASE_CONFIG=$CONFIG_BASE_DIR/classifiers/$CLASSIFIER_NAME.yml
DATASET_SPECIFIC_OVERRIDE=$CONFIG_BASE_DIR/datasets/$DATASET/_.yml
if [ -f $DATASET_SPECIFIC_OVERRIDE ]; then
    BASE_CONFIG="$BASE_CONFIG $DATASET_SPECIFIC_OVERRIDE"
fi
DATASET_CLASSIFIER_SPECIFIC_OVERRIDE=$CONFIG_BASE_DIR/datasets/$DATASET/$CLASSIFIER_NAME.yml
if [ -f $DATASET_CLASSIFIER_SPECIFIC_OVERRIDE ]; then
    BASE_CONFIG="$BASE_CONFIG $DATASET_CLASSIFIER_SPECIFIC_OVERRIDE"
fi

python -m active_learning_lab.run_experiment \
    $EXPERIMENT_NAME \
    $CONFIG_BASE_DIR/base.yml $BASE_CONFIG \
    --dataset_name=$DATASET --query_strategy=$QUERY_STRATEGY "${@:5}" \
    --max-reproducibility
