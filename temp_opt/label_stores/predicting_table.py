import torch

from typing import Dict


class PredictingTable:

    def __init__(self, predicting_dict: Dict[torch.nn.Module, torch.utils.data.DataLoader]):
        """
        :param predicting_dict: [module model, dataloader valid_dataloader]
        """
        self._predicting_dict = predicting_dict
        self._models = list(self._predicting_dict.keys())
        self._max_len = len(predicting_dict)
        self._index = 0

    def __len__(self):
        return self._max_len

    def __iter__(self):
        return self

    def __next__(self):
        if self._max_len <= self._index:
            self._index = 0
            raise StopIteration()
        model = self._models[self._index]
        val_loader = self._predicting_dict[model]
        self._index += 1
        return model, val_loader
