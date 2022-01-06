import os
from typing import Optional, List

import pandas as pd

from constants import *


class H2OParser:
    def __init__(self):
        self._lines: Optional[List[str]] = None
        self._line_idx: int = 0

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value
        self._line_idx = 0

    def continue_to_line_with(self, substring: str) -> str:
        while substring not in self._lines[self._line_idx]:
            self._line_idx += 1
        return self._lines[self._line_idx]

    def create_cv_summary(self):
        """
        Write a summary of all the CV results.
        """
        rows = []
        for dataset_name in os.listdir(DATASETS):
            output_dir = f'{DATASETS}/{dataset_name}/{OUTPUT}'
            for version in os.listdir(output_dir):
                with open(f'{output_dir}/{version}/{CV_RESULTS}.txt') as f:
                    self.lines = f.readlines()
                    self.continue_to_line_with('** Reported on cross-validation data. **')
                    model = self._lines[self._line_idx - 1][:-1]
                    row = [dataset_name, version.rpartition(".")[0], model]
                    line = self.continue_to_line_with('AUC: ')
                    row.append(float(line.partition(' ')[-1]))
                    matrix_info = self.continue_to_line_with('Confusion Matrix (Act/Pred)')
                    threshold = float(matrix_info.split(' ')[-1][:-2])
                    matrix_header = self._lines[self._line_idx + 1].split('\t')[:-2]
                    matrix_row_1 = [float(i) for i in self._lines[self._line_idx + 2].split('\t')[2:-2]]
                    matrix_row_2 = [float(i) for i in self._lines[self._line_idx + 3].split('\t')[2:-2]]
                    row.extend(self.parse_matrix(matrix_header, [matrix_row_1, matrix_row_2]))
                    row.append(threshold)
                    rows.append(row)
        header = [
            'dataset',
            'version',
            'model',
            'AUC',
            'accuracy',
            'confirmed precision',
            'confirmed recall',
            'rejected precision',
            'rejected recall',
            'threshold',
        ]
        if not os.path.exists(CV_SUMMARIES):
            os.makedirs(CV_SUMMARIES)
        n_summaries = len(os.listdir(CV_SUMMARIES))
        pd.DataFrame(rows, columns=header).to_csv(f'cv_summary_{n_summaries + 1}.csv', index=False)

    @staticmethod
    def parse_matrix(header, matrix: list) -> tuple:
        idx_conf, idx_rej = (0, 1) if header[0] == CONFIRMED else (1, 0)
        total = sum(sum(r) for r in matrix)
        n_true_conf = sum(matrix[idx_conf])
        n_true_rej = sum(matrix[idx_rej])
        n_pred_conf = sum(r[idx_conf] for r in matrix)
        n_pred_rej = sum(r[idx_rej] for r in matrix)
        n_true_positive_conf = matrix[idx_conf][idx_conf]
        n_true_positive_rej = matrix[idx_rej][idx_rej]
        return (
            (n_true_positive_conf + n_true_positive_rej) / total,  # accuracy
            n_true_positive_conf / n_pred_conf,  # confirmed precision
            n_true_positive_conf / n_true_conf,  # confirmed recall
            n_true_positive_rej / n_pred_rej,  # rejected precision
            n_true_positive_rej / n_true_rej,  # rejected recall
        )


if __name__ == '__main__':
    H2OParser().create_cv_summary()
