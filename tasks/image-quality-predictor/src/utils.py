from typing import Union

import torch


class EarlyStopping:

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self._counter = 0

    def __call__(self, bestLoss: Union[float, torch.Tensor], latestLoss: Union[float, torch.Tensor]) -> bool:
        if latestLoss >= bestLoss:
            # Loss did not improve
            self._counter += 1
        else:
            # Loss improved
            self._counter = 0

        return self._counter >= self.patience
