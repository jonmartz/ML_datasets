import json
import subprocess

import numpy as np
import pandas as pd

from common_utils import evaluate_predictions


def test():
    model_dir = f'datasets/{train_dataset}/output/{train_version}/model'
    with open(f'{model_dir}/metadata.json') as f:
        model_metadata = json.load(f)
    dataset = pd.read_csv(test_filename).rename(columns={v: k for k, v in model_metadata['features'].items()})
    if 'GROUP' in dataset:
        dataset = dataset.drop(columns='GROUP')
    dataset.drop(columns='status').to_csv(f'{model_dir}/tmp/dataset.csv', index=False)
    command = f'java -cp .:h2o-genmodel.jar main model.zip False'
    output = subprocess.run(command, cwd=model_dir, shell=True, capture_output=True)
    if output.stderr:
        raise Exception(f'error running H2O model: {output.stderr}')
    predictions = pd.read_csv(f'{model_dir}/tmp/predictions.csv')
    evaluate_predictions(dataset['status'], predictions['prediction'])


if __name__ == '__main__':
    train_dataset = 'ml_features'
    train_version = 'filled_unbalanced'
    test_filename = 'test_ml_features_dataset_filled.csv'

    test()

