import json

import numpy as np
import pandas as pd
from ml_utils.classification_manager import ClassificationManager

from common_utils import evaluate_predictions


def test():
    with open(f'inferencers/{model_name}.json') as f:
        params = json.load(f)
    model = ClassificationManager.load_saved_model(params)
    dataset = pd.read_csv(test_filename)
    predictions = []
    n = len(dataset)
    for i, r in dataset.iterrows():
        predictions.append(model.predict(r)[0])
        print(f'{(i+1)/n*100:.2f}%')
    print()
    evaluate_predictions(dataset['status'], np.array(predictions))


if __name__ == '__main__':
    model_name = 'ML6'
    test_filename = 'test_ml_features_dataset_filled.csv'

    test()
