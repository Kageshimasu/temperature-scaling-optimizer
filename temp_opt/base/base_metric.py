import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseMetric(nn.Module):

    def __init__(self):
        super(BaseMetric, self).__init__()

    @abstractmethod
    def forward(self, logits: torch.Tensor or np.ndarray, labels: torch.Tensor or np.ndarray):
        pass
