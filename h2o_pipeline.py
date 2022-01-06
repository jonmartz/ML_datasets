from argparse import ArgumentParser
from typing import List

from constants import *

import h2o
from h2o.automl import H2OAutoML


def run_h2o(dataset_name: str, include_algos: List[str] = None, max_models: int = 20, balance_classes: bool = False):
    h2o.init()

    # load dataset
    dataset_path = f'{DATASETS}/{dataset_name}/dataset.csv'
    dataset = h2o.import_file(dataset_path, header=1)
    x = dataset.columns
    x.remove(TARGET_CLASS_NAME)
    dataset[TARGET_CLASS_NAME] = dataset[TARGET_CLASS_NAME].asfactor()

    # run AutoML
    aml = H2OAutoML(include_algos=include_algos, max_models=max_models, seed=1, balance_classes=balance_classes)
    aml.train(x=x, y=TARGET_CLASS_NAME, training_frame=dataset, fold_column='FOLD')

    # View the AutoML Leaderboard
    lb = aml.leaderboard
    lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

    # The leader model is stored here
    print(f'leader: {aml.leader}')

if __name__ == '__main__':
    parser = ArgumentParser(description='Generic ML pipeline')
    parser.add_argument('--dataset', dest='dataset_name', type=str, required=True,
                        help='Name of training dataset (name of dir located in "datasets" dir)')
    run_h2o()