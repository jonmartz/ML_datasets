import os
from argparse import ArgumentParser
# import logging

import h2o
from h2o.automl import H2OAutoML
# from cybmodules.tools.logging import logmanager

from constants import *

# logmanager.setup()
# logger = logging.getLogger(__name__)

valid_algos = {
    'DRF',
    'GLM',
    'XGBoost',
    'GBM',
    'DeepLearning',
    'StackedEnsemble',
}


def run_h2o(args):
    if not os.path.exists(TMP_LOG):
        os.makedirs(TMP_LOG)
    h2o.init(log_dir=TMP_LOG, verbose=False)

    # create output dirs and initialize h2o cluster
    version = ('un' if args.has_missing else '') + 'filled_' + ('' if args.balance_classes else 'un') + 'balanced'
    output_dir = f'{DATASETS}/{args.dataset_name}/{OUTPUT}/{version}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # h2o.init()

    # load dataset
    dataset_path = f'{DATASETS}/{args.dataset_name}/dataset.csv'
    dataset = h2o.import_file(dataset_path, header=1)
    x = dataset.columns[1:-1]
    dataset[TARGET_CLASS_NAME] = dataset[TARGET_CLASS_NAME].asfactor()

    # get best model
    auto_ml = H2OAutoML(include_algos=args.include_algos, max_models=args.max_models,
                        balance_classes=args.balance_classes, seed=1)
    auto_ml.train(x=x, y=TARGET_CLASS_NAME, training_frame=dataset, fold_column='FOLD')
    best_model = auto_ml.leader
    best_model.save_mojo(f'{output_dir}/best_model')

    # View the AutoML Leaderboard
    # lb = auto_ml.leaderboard
    # lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

    for filename in os.listdir(TMP_LOG):  # clean tmp log
        os.unlink(f'{TMP_LOG}/{filename}')

    print('\ndone')


if __name__ == '__main__':
    parser = ArgumentParser(description='H2O AutoML Pipeline')
    parser.add_argument('-d', dest='dataset_name', type=str, required=True,
                        help='Name of training dataset (name of dir located in "datasets" dir)')
    parser.add_argument('-include_algos', nargs='+', default=None,
                        help=f'List of algorithm types to consider, from {sorted(valid_algos)}. Default: All of them.')
    parser.add_argument('-max_models', dest='max_models', type=int, default=20,
                        help='Maximum amount of models to consider. Default: 20.')
    parser.add_argument('-balance_classes', action='store_true', help='Whether to perform class balancing.')
    parser.add_argument('-has_missing', action='store_true', help='Whether the dataset contains missing values.')

    args = parser.parse_args()

    for model_type in args.include_algos or []:
        if model_type not in valid_algos:
            raise ValueError(f'"{model_type}" is not one of the valid model types: {sorted(valid_algos)}')

    run_h2o(args)
