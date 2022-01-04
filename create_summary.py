import os
from typing import Iterator, Optional, List

import pandas as pd

RESULTS_DIR = 'cv_results'

CONFIRMED = 'Confirmed'
REJECTED = 'Rejected'

class Parser:
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

    def parse(self):
        rows = []
        for version in os.listdir(RESULTS_DIR):
            version_dir = f'{RESULTS_DIR}/{version}'
            for sub_version in os.listdir(version_dir):
                with open(f'{version_dir}/{sub_version}') as f:
                    row = [f'{version} - {sub_version.rpartition(".")[0]}']
                    self.lines = f.readlines()
                    self.continue_to_line_with('** Reported on cross-validation data. **')
                    model = self._lines[self._line_idx - 1][:-1]
                    line = self.continue_to_line_with('AUC: ')
                    row.append(float(line.partition(' ')[-1]))
                    matrix_info = self.continue_to_line_with('Confusion Matrix (Act/Pred)')
                    threshold = float(matrix_info.split(' ')[-1][:-2])
                    matrix_header = self._lines[self._line_idx + 1].split('\t')[:-2]
                    matrix_row_1 = [float(i) for i in self._lines[self._line_idx + 2].split('\t')[2:-2]]
                    matrix_row_2 = [float(i) for i in self._lines[self._line_idx + 3].split('\t')[2:-2]]
                    row.extend(self.parse_matrix(matrix_header, [matrix_row_1, matrix_row_2]))
                    row.extend((threshold, model))
                    rows.append(row)
        header = [
            'dataset version',
            'AUC',
            'accuracy',
            'confirmed precision',
            'confirmed recall',
            'rejected precision',
            'rejected recall',
            'threshold',
            'model',
        ]
        pd.DataFrame(rows, columns=header).to_csv('report.csv', index=False)

    def parse_matrix(self, header, matrix: list) -> tuple:
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
    Parser().parse()
